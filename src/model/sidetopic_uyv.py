#!/usr/bin/python
# -*- coding: utf-8 -*- 

'''
The inputs and outputs of a SideTopicModel

Created on 29 Jun 2013

@author: bryanfeeney
'''

from math import log
from math import pi
from math import e
from collections import namedtuple
import numpy as np
import scipy.linalg as la
import scipy.sparse as ssp
import scipy.sparse.linalg as sla
import numpy.random as rd
import matplotlib.pyplot as plt

from util.overflow_safe import safe_log, safe_x_log_x, safe_log_one_plus_exp_of
from util.array_utils import normalizerows_ip, rowwise_softmax

from numba import autojit

# TODO Consider using numba for autojit (And jit with local types)
# TODO Investigate numba structs as an alternative to namedtuples
# TODO Make random() stuff predictable, either by incorporating a RandomState instance into model parameters
#      or calling a global rd.seed(0xC0FFEE) call.
# TODO Sigma and Tau optimisation is hugely expensive, not only because of their own updates,
#      but because were they fixed, we wouldn't need to do any updates for varA, which would save 
#      us from doing a FxF inverse at every iteration. 
# TODO varA is a huge, likely dense, FxF matrix
# TODO varV is a big, dense, PxP matrix...
# TODO Storing the vocab twice (vocab and lnVocab) is expensive
# TODO How slow is safe_log?
# TODO Eventually s just overflows
# TODO Sigma update causes NaNs in the variational-bound

# ==============================================================
# CONSTANTS
# ==============================================================

MAX_X_TICKS_PER_PLOT = 50
DTYPE = np.float32

LOG_2PI   = log(2 * pi)
LOG_2PI_E = log(2 * pi * e)

# ==============================================================
# TUPLES
# ==============================================================

VbSideTopicQueryState = namedtuple ( \
    'VbSideTopicState', \
    'lmda nu lxi s docLen'\
)


VbSideTopicModelState = namedtuple ( \
    'VbSideTopicState', \
    'K, Q, F, P, T, A, varA, Y, omY, sigY, U, V, vocab, tau, sigma'
)

# ==============================================================
# CODE
# ==============================================================

def negJakkola(vec):
    '''
    The negated version of the Jakkola expression which was used in Bourchard's NIPS
    2007 softmax bound
    
    CTM Source reads: y = .5./x.*(1./(1+exp(-x)) -.5);
    '''
    
    # COPY AND PASTE BETWEEN THIS AND negJakkolaOfDerivedXi()
    return 0.5/vec * (1./(1 + np.exp(-vec)) - 0.5)

def negJakkolaOfDerivedXi(lmda, nu, s, d = None):
    '''
    The negated version of the Jakkola expression which was used in Bouchard's NIPS '07
    softmax bound calculated using an estimate of xi derived from lambda, nu, and s
    
    expLmda - the DxK matrix of means of e to the power of the topic distribution for each document.
              Note that this is different to all other implementations of neg-Jaakkola which assume
              the topic distribution is provided, and not e to the power of it.
    nu      - the DxK the vector of variances of the topic distribution
    s       - The Dx1 vector of offsets.
    d       - the document index (for lambda and nu). If not specified we construct
              the full matrix of A(xi_dk)
    '''
    
    # COPY AND PASTE BETWEEN THIS AND negJakkola()
    if d is not None:
        vec = (np.sqrt (lmda[d,:]**2 - 2 * lmda[d,:] * s[d] + s[d]**2 + nu[d,:]**2))
        return 0.5/vec * (1./(1 + np.exp(-vec)) - 0.5)
    else:
        mat = deriveXi(lmda, nu, s)
        return 0.5/mat * (1./(1 + np.exp(-mat)) - 0.5)
    

def jakkolaOfDerivedXi(lmda, nu, s, d = None):
    '''
    The standard version of the Jakkola expression which was used in Bouchard's NIPS '07
    softmax bound calculated using an estimate of xi derived from lambda, nu, and s
    
    lmda - the DxK matrix of means of the topic distribution for each document
    nu   - the DxK the vector of variances of the topic distribution
    s    - The Dx1 vector of offsets.
    d    - the document index (for lambda and nu). If not specified we construct
           the full matrix of A(xi_dk)
    '''
    
    # COPY AND PASTE BETWEEN THIS AND negJakkola()
    if d is not None:
        vec = (np.sqrt (lmda[d,:]**2 -2 *lmda[d,:] * s[d] + s[d]**2 + nu[d,:]**2))
        return 0.5/vec * (1./(1 + np.exp(-vec)) - 0.5)
    else:
        mat = deriveXi(lmda, nu, s)
        return 0.5/mat * (0.5 - 1./(1 + np.exp(-mat)))


def train(modelState, X, W, iterations=10000, epsilon=0.001, logInterval = 0, plotInterval = 0):
    '''
    Creates a new query state object for a topic model based on side-information. 
    This contains all those estimated parameters that are specific to the actual
    date being queried - this must be used in conjunction with a model state.
    
    The parameters are
    
    modelState - the model state with all the model parameters
    X - the D x F matrix of side information vectors
    W - the D x V matrix of word **count** vectors.
    iterations - how long to iterate for
    epsilon - currently ignored, in future, allows us to stop early.
    
    This returns a tuple of new model-state and query-state. The latter object will
    contain X and W and also
    
    s      - A D-dimensional vector describing the offset in our bound on the true value of ln sum_k e^theta_dk 
    lxi    - A DxK matrix used in the above bound, containing the negative Jakkola function applied to the 
             quadratic term xi
    lambda - the topics we've inferred for the current batch of documents
    nu     - the variance of topics we've inferred (independent)
    '''
    # Unpack the model state tuple for ease of use and maybe speed improvements
    (K, Q, F, P, T, A, omA, Y, omY, sigY, U, V, vocab, tau, sigma) = (modelState.K, modelState.Q, modelState.F, modelState.P, modelState.T, modelState.A, modelState.varA, modelState.Y, modelState.omY, modelState.sigY, modelState.U, modelState.V, modelState.vocab, modelState.tau, modelState.sigma)
    
    if W.dtype.kind == 'i':
        W = W.astype(np.float32)
    
    # Get ready to plot the evolution of the likelihood
    if logInterval > 0:
        elbos = np.zeros((iterations / logInterval,))
        iters = np.zeros((iterations / logInterval,))
    iters.fill(-1)
    
    # We'll need the total word count per doc, and total count of docs
    docLen = np.squeeze(np.asarray (W.sum(axis=1))) # Force to a one-dimensional array for np.newaxis trick to work
    D      = len(docLen)
    
    # No need to recompute this every time
    XTX = X.T.dot(X)
    
    # Identity matrices that occur
    I_P  = np.eye(P,P, 0, DTYPE)
    I_Q  = np.eye(Q,Q, 0, DTYPE)
    I_F  = ssp.eye(F,F, 0, DTYPE, "csc") # X is CSR, XTX is consequently CSC, sparse inverse requires CSC
    
    # Assign initial values to the query parameters
    expLmda = np.exp(rd.random((D, K)).astype(DTYPE))
    nu   = np.ones((D, K), DTYPE)
    s    = np.zeros((D,), DTYPE)
    lxi  = negJakkola (np.ones((D,K), DTYPE))
    
    # If we don't bother optimising either tau or sigma we can just do all this here once only 
    tsq     = tau * tau;
    ssq     = sigma * sigma;
    overTsq = 1. / tsq
    overSsq = 1. / ssq
    overTsqSsq = 1./(tsq * ssq)
    
    # TODO the inverse being almost always dense means that it might
    # be faster to convert to dense and use the normal solver, despite
    # the size constraints.
#    varA = 1./K * sla.inv (overTsq * I_F + overSsq * XTX)
    tI_sXTX = (overTsq * I_F + overSsq * XTX).todense(); 
    omA = la.inv (tI_sXTX)
    scaledWordCounts = W.copy()
   
    for iteration in range(iterations):
        
        # =============================================================
        # E-Step
        #   Model dists are q(Theta|A;Lambda;nu) q(A|Y) q(Y) and q(Z)....
        #   Where lambda is the posterior mean of theta.
        # =============================================================
              
      
        # Y, sigY, omY
        # 
        VTV = V.T.dot(V)
        UTU = U.T.dot(U)
        try:
            invUTU = la.inv(UTU)                  # [ should we experiment with chol decomp? And why is it singular? ]
        except np.linalg.linalg.LinAlgError as e: # U seems to rapidly become singular (before 5 iters) Need to add a switch 
            invUTU = la.pinvh(UTU)                # for this, it causes the monotonically increasing VB bound to decrease!
                                  
        
        # The update for Y is 
        #    ssq * tsq * Y + UTU.dot(Y).dot(VTV) = U.T.dot(A).V
        # If we pre-multiply by la.inv(UTU) (the smallest matrix going) we use the built in solve
        #    ssq * tsq * la.inv(UTU) * Y + Y.dot(VTV) = la.inv(UTU).dot(U.T).dot(A).dot(V)
        # However as UTU rapidly becomes non-singular, we use the pseudo inverse, which isn't great...
        Y = la.solve_sylvester (ssq * tsq * invUTU, VTV, invUTU.dot(U.T).dot(A).dot(V)) 
        _quickPrintElbo ("E-Step: q(Y) [Mean]", iteration, X, W, K, Q, F, P, T, A, omA, Y, omY, sigY, U, V, vocab, tau, sigma, expLmda, nu, lxi, s, docLen)
        
        sigY = 1./P * la.inv(np.trace(omY)  * I_Q + overTsqSsq * np.trace(omY.dot(VTV)) * UTU)
        _quickPrintElbo ("E-Step: q(Y) [sigY]", iteration, X, W, K, Q, F, P, T, A, omA, Y, omY, sigY, U, V, vocab, tau, sigma, expLmda, nu, lxi, s, docLen)
        
        omY  = 1./Q * la.inv(np.trace(sigY) * I_P + overTsqSsq * np.trace(sigY.dot(UTU)) * VTV) 
        _quickPrintElbo ("E-Step: q(Y) [omY]", iteration, X, W, K, Q, F, P, T, A, omA, Y, omY, sigY, U, V, vocab, tau, sigma, expLmda, nu, lxi, s, docLen)
        
        # A 
        #
        # So it's normally A = (UYV' + L'X) omA with omA = inv(t*I_F + s*XTX)
        #   so A inv(omA) = UYV' + L'X
        #   so inv(omA)' A' = VY'U' + X'L
        # at which point we can use a built-in solve
        #
#       A = (overTsq * U.dot(Y).dot(V.T) + X.T.dot(expLmda).T).dot(omA)
        A = la.solve(tI_sXTX, X.T.dot(expLmda) + V.dot(Y.T).dot(U.T)).T
        _quickPrintElbo ("E-Step: q(A)", iteration, X, W, K, Q, F, P, T, A, omA, Y, omY, sigY, U, V, vocab, tau, sigma, expLmda, nu, lxi, s, docLen)
       
        # lmda_dk
        #
        
        # There are many more opportunities to optimise the expression: sc = W / lmda.dot(vocab)
        #  e.g. Using the indices to evaluate the individual dot products sc[r,c] = w[r,c] / sum_k l[r,k] v[k,c]
        #  without fully materialising the dense DxT dot product
        scaledWordCounts.data[:] = W.data
        scaledWordCounts.data /= expLmda.dot(vocab)[csr_indices(scaledWordCounts.indptr, scaledWordCounts.indices)]
        
        XAT = X.dot(A.T)
        rho = 2 * s[:,np.newaxis] * lxi - 0.5 \
            + expLmda * (scaledWordCounts.dot(vocab.T)) / docLen[:,np.newaxis]  
        rhs  = docLen[:,np.newaxis] * rho + overSsq * XAT
        
        expLmda = rhs / (docLen[:,np.newaxis] * 2 * lxi + overSsq)
        # Note we haven't applied np.exp() yet, we're holding off till we've evaluated the next few terms
        # This efficiency saving only actually applies once we've disabled all the _quickPrintElbo calls
        
        _quickPrintElbo ("E-Step: q(\u03F4) [Mean]", iteration, X, W, K, Q, F, P, T, A, omA, Y, omY, sigY, U, V, vocab, tau, sigma, np.exp(expLmda), nu, lxi, s, docLen)
         
        # nu_dk
        #
        nu = 1./ np.sqrt(2. * docLen[:, np.newaxis] * lxi + overSsq)
        _quickPrintElbo ("E-Step: q(\u03F4) [Var] ", iteration, X, W, K, Q, F, P, T, A, omA, Y, omY, sigY, U, V, vocab, tau, sigma, np.exp(expLmda), nu, lxi, s, docLen)
          
        # s_d
        #
        s = (K/4. - 0.5 + (lxi * expLmda).sum(axis = 1)) / lxi.sum(axis=1)
        _quickPrintElbo ("E-Step: s_d", iteration, X, W, K, Q, F, P, T, A, omA, Y, omY, sigY, U, V, vocab, tau, sigma, np.exp(expLmda), nu, lxi, s, docLen)
        
        # xi_dk
        # 
        lxi = negJakkolaOfDerivedXi(expLmda, nu, s)
        _quickPrintElbo ("E-Step: \u039B(xi_dk)", iteration, X, W, K, Q, F, P, T, A, omA, Y, omY, sigY, U, V, vocab, tau, sigma, np.exp(expLmda), nu, lxi, s, docLen)

        # Now finally we finish off the estimate of exp(lmda)
        np.exp(expLmda, out=expLmda)
       
        # =============================================================
        # M-Step
        #    Parameters for the softmax bound: lxi and s <-- ?
        #    The projection used for A: U and V
        #    The vocabulary : vocab
        #    The variances: tau, sigma
        # =============================================================
        
               
        # U
        # TODO Verify this...
        U = A.dot(V).dot(Y.T).dot (la.inv(Y.dot(V.T).dot(V).dot(Y.T) + np.trace(omY.dot(V.T).dot(V)) * sigY))
        _quickPrintElbo ("M-Step: U", iteration, X, W, K, Q, F, P, T, A, omA, Y, omY, sigY, U, V, vocab, tau, sigma, expLmda, nu, lxi, s, docLen)

        # V
        # 
        V = A.T.dot(U).dot(Y).dot (la.inv(Y.T.dot(U.T).dot(U).dot(Y) + np.trace(sigY.dot(U.T).dot(U)) * omY))
        _quickPrintElbo ("M-Step: V", iteration, X, W, K, Q, F, P, T, A, omA, Y, omY, sigY, U, V, vocab, tau, sigma, expLmda, nu, lxi, s, docLen)

        #
        # vocab
        #
        factor = (scaledWordCounts.T.dot(expLmda)).T # Gets materialized as a dense matrix...
        vocab *= factor
        normalizerows_ip(vocab)
        _quickPrintElbo ("M-Step: \u03A6", iteration, X, W, K, Q, F, P, T, A, omA, Y, omY, sigY, U, V, vocab, tau, sigma, expLmda, nu, lxi, s, docLen)
        
        if (logInterval > 0) and (iteration % logInterval == 0):
            np.log(expLmda, out=expLmda)
            elbo = varBound ( \
                VbSideTopicModelState (K, Q, F, P, T, A, omA, Y, omY, sigY, U, V, vocab, tau, sigma), \
                VbSideTopicQueryState(expLmda, nu, lxi, s, docLen),
                X, W, None, XAT, XTX)
            
            np.exp(expLmda, out=expLmda)
                
            elbos[iteration / logInterval] = elbo
            iters[iteration / logInterval] = iteration
            print ("Iteration %5d  ELBO %f" % (iteration, elbo))
        
        if (plotInterval > 0) and (iteration % plotInterval == 0) and (iteration > 0):
            plot_bound(iters, elbos)
            
        if (iteration % 10 == 0) and (iteration > 0):
            print ("\n\nOmega_Y[0,:] = " + str(omY[0,:]))
            print ("Sigma_Y[0,:] = " + str(sigY[0,:]))
            
    
    if plotInterval > 0:
        plot_bound(iters, elbos)
    
    return (VbSideTopicModelState (K, Q, F, P, T, A, omA, Y, omY, sigY, U, V, vocab, tau, sigma), \
            VbSideTopicQueryState (np.log(expLmda, out=expLmda), nu, lxi, s, docLen))
    
def plot_bound (iters, bounds):
    '''
    Plots the evolution of the variational bound. The input is a pair of
    matched arrays: for a given point i, iters[i] was the iteration at which
    the bound bounds[i] was calculated
    '''
    itersFilled = [i for i in iters if i >= 0]
    numValues   = len(itersFilled)
    boundsFilled = bounds if numValues == len(iters) else bounds[:numValues]
    
    fig  = plt.figure()
    plot = fig.add_subplot(1,1,1)
    plot.plot (itersFilled, boundsFilled)
    plot.set_ylabel("Bound")
    plot.set_xlabel("Iteration")
    plot.set_xticks(itersFilled)
    plt.show()
    
def _quickPrintElbo (updateMsg, iteration, X, W, K, Q, F, P, T, A, varA, Y, omY, sigY, U, V, vocab, tau, sigma, expLmda, nu, lxi, s, docLen):
    '''
    Calculates the variational lower bound and prints it to stdout,
    prefixed with a table and the given updateMsg
    
    See varBound() for a full description of all parameters
    
    Obviously this is a very ugly inefficient method.
    '''
#     if iteration % 100 != 0:
#         return
    
    lmda = np.log(expLmda)
    xi = deriveXi(lmda, nu, s)
    elbo = varBound ( \
                      VbSideTopicModelState (K, Q, F, P, T, A, varA, Y, omY, sigY, U, V, vocab, tau, sigma), \
                      VbSideTopicQueryState(lmda, nu, lxi, s, docLen), \
                      X, W)
    
    
    print ("\t Update %-30s  ELBO : %12.3f  lmda.mean=%f \tlmda.max=%f \tlmda.min=%f \tnu.mean=%f \txi.mean=%f \ts.mean=%f" % (updateMsg, elbo, lmda.mean(), lmda.max(), lmda.min(), nu.mean(), xi.mean(), s.mean()))

def varBound (modelState, queryState, X, W, lnVocab = None, XAT=None, XTX = None, scaledWordCounts = None):
    '''
    For a current state of the model, and the query, for given inputs, outputs the variational
    lower-bound.
    
    Params
    
    modelState - the state of the model currently
    queryState - the state of the query currently
    X          - the DxF matrix of features we're querying on, where D is the number of documents
    W          - the DxT matrix of words ("terms") we're querying on
    Z          - if this has already been calculated, it can be passed in. If not, we
                 recalculate it from the model and query states. Z is the DxKxT tensor which
                 for each document D and term T gives the proportion of those terms assigned
                 to topic K
    vocab      - the KxV matrix of the vocabulary distribution
    XAT        - DxK dot product of XA', recalculated if not provided, where X is DxF and A' is FxK
    XTX        - dot product of X-transpose and X, recalculated if not provided.
    
    Returns
        The (positive) variational lower bound
    '''
    
    # Unpack the model and query state tuples for ease of use and maybe speed improvements
    (K, Q, F, P, T, A, omA, Y, omY, sigY, U, V, vocab, tau, sigma) = (modelState.K, modelState.Q, modelState.F, modelState.P, modelState.T, modelState.A, modelState.varA, modelState.Y, modelState.omY, modelState.sigY, modelState.U, modelState.V, modelState.vocab, modelState.tau, modelState.sigma)
    (lmda, nu, lxi, s, docLen) = (queryState.lmda, queryState.nu, queryState.lxi, queryState.s, queryState.docLen)
    
    # Get the number of samples from the shape. Ensure that the shapes are consistent
    # with the model parameters.
    (D, Tcheck) = W.shape
    if Tcheck != T: raise ValueError ("The shape of the DxT document matrix W is invalid, T is %d but the matrix W has shape (%d, %d)" % (T, D, Tcheck))
    
    (Dcheck, Fcheck) = X.shape
    if Dcheck != D: raise ValueError ("Inconsistent sizes between the matrices X and W, X has %d rows but W has %d" % (Dcheck, D))
    if Fcheck != F: raise ValueError ("The shape of the DxF feature matrix X is invalid. F is %d but the matrix X has shape (%d, %d)" % (F, Dcheck, Fcheck)) 

    # We'll need the original xi for this and also Z, the 3D tensor of which for each document D 
    # and term T gives the strength of topic K. We'll also need the log of the vocab dist
    xi = deriveXi (lmda, nu, s)
    
    # If not already provided, we'll also need the the product of XA
    #
    if XAT is None:
        XAT = X.dot(A.T)
    if XTX is None:
        XTX = X.T.dot(X)
   
    # <ln p(Y)>
    # 
    lnP_Y = -0.5 * (Q*P * LOG_2PI + np.trace(sigY) * np.trace(omY) + np.sum(Y * Y))
    
    # <ln P(A|Y)>
    # TODO it looks like I should take the trace of omA \otimes I_K here.
    # TODO Need to check re-arranging sigY and omY is sensible.
    halfKF = 0.5 * K * F
    halfTsq = 0.5 / (tau * tau)
    lnP_A = -halfKF * LOG_2PI - halfKF * log (tau * tau) \
            -halfTsq * (np.sum(omY * V.T.dot(V)) * np.sum(sigY * U.T.dot(U)) \
                      + np.trace(XTX.dot(omA)) * K \
                      + np.sum (np.square(A - U.dot(Y).dot(V.T))))
    # <ln p(Theta|A,X)
    # 
    sig2  = sigma * sigma
    tau2  = tau * tau
    
    lnP_Theta = -0.5 * D * LOG_2PI -0.5 * D * K * log (sig2) \
                - 0.5 / sig2 * ( \
                    np.sum(nu) + D*K * tau2 * np.sum(XTX * omA) + np.sum(np.square(lmda - XAT)))
    
    # <ln p(Z|Theta)
    # 
    docLenLmdaLxi = docLen[:, np.newaxis] * lmda * lxi
    expLmda = np.exp(lmda)
    if scaledWordCounts is None:
        scaledWordCounts = W.copy()
        scaledWordCounts.data /= expLmda.dot(vocab)[csr_indices(scaledWordCounts.indptr, scaledWordCounts.indices)]

    lnP_Z = 0.0
    lnP_Z -= np.sum(docLenLmdaLxi * lmda)
    lnP_Z -= np.sum(docLen[:, np.newaxis] * nu * nu * lxi)
    lnP_Z += 2 * np.sum (s[:, np.newaxis] * docLenLmdaLxi)
    lnP_Z -= 0.5 * np.sum (docLen[:, np.newaxis] * lmda)
    lnP_Z += np.sum (lmda * expLmda * (scaledWordCounts.dot(vocab.T))) # n(d,k) = expLmda * (scaledWordCounts.dot(vocab.T))
    lnP_Z -= np.sum(docLen[:,np.newaxis] * lxi * ((s**2)[:,np.newaxis] - xi**2))
    lnP_Z += 0.5 * np.sum(docLen[:,np.newaxis] * (s[:,np.newaxis] + xi))
    lnP_Z -= np.sum(docLen[:,np.newaxis] * safe_log_one_plus_exp_of(xi))
    lnP_Z -= np.sum (docLen * s)
        
    # <ln p(W|Z, vocab)>
    # 
    lnP_w_dt = scaledWordCounts.copy() # Probability matrix of every term t in every individual document d
    lnP_w_dt.data *= (expLmda.dot(vocab * safe_log(vocab)))[csr_indices(lnP_w_dt.indptr, lnP_w_dt.indices)]
    lnP_W = np.sum(lnP_w_dt.data)
    
    # H[q(Y)]
    ent_Y = 0.5 * (P * K * LOG_2PI_E + Q * log (la.det(omY)) + P * log (la.det(sigY)))
    
    # H[q(A|Y)]
    ent_A = 0.5 * (F * K * LOG_2PI_E + K * log (la.det(omA)) + F * K * log (tau2))
    
    # H[q(Theta|A)]
    ent_Theta = 0.5 * (K * LOG_2PI_E + np.sum (np.log(nu * nu)))
    
    # H[q(Z|\Theta)
    #
    # So Z_dtk \propto expLmda_dt * vocab_tk. We let N here be the normalizer (which is 
    # \sum_j expLmda_dt * vocab_tj, which implies N is DxT. We need to evaluate
    # Z_dtk * log Z_dtk. We can pull out the normalizer of the first term, but it has
    # to stay in the log Z_dtk expression, hence the third term in the sum. We can however
    # take advantage of the ability to mix dot and element-wise products for the different
    # components of Z_dtk in that three-term sum, which we denote as S
    #   Finally we use np.sum to sum over d and t
    #
    N = expLmda.dot(vocab) + 1E-35 # DxT !!! TODO Figure out why this is zero sometimes (better init of vocab?)
    S = expLmda.dot(vocab * safe_log(vocab)) + (expLmda * np.log(expLmda)).dot(vocab) - N * safe_log(N)
    np.reciprocal(N, out=N)
    ent_Z = -np.sum (N * S)
    
    result = lnP_Y + lnP_A + lnP_Theta + lnP_Z + lnP_W + ent_Y + ent_A + ent_Theta + ent_Z
#    if (lnP_Z > 0) or (lnP_Theta > 0) or (lnProb3 > 0) or (lnProb4 > 0):
#        print ("Whoopsie - lnProb > 0")
    
#    if result > 100:
#        print ("Well this is just ridiculous")
    
    return result
 
def deriveXi (lmda, nu, s):
    '''
    Derives a value for xi. This is not normally needed directly, as we
    normally just work with the negJakkola() function of it
    '''
    return np.sqrt(lmda**2 - 2 * lmda * s[:,np.newaxis] + (s**2)[:,np.newaxis] + nu**2)   


def newVbModelState(K, Q, F, P, T):
    '''
    Creates a new model state object for a topic model based on side-information. This state
    contains all parameters that once trained can be kept fixed for querying.
    
    The parameters are
    
    K - the number of topics
    Q - the number of latent topics, Q << K
    F - the number of features
    P - the number of latent features in the projected space, P << F
    T - the number of terms in the vocabulary
    
    The returned object will contain K, Q, F, P and T and also
    
    A      - the mean of the KxF matrix mapping F features to K topics. 
    varA   - a vector containing the variance over the F features of the distribution over A
    Y      - the latent space which is mixed by U and V into the observed space
    omY    - the row variance of the distribution over Y
    sigY   - the column variance of the distribution over Y
    U      - the KxQ transformation from the K dimensional observed topic space to the
             Q-dimensional topic space
    V      - the FxP transformation from the F-dimensinal observed features space to the
             latent P-dimensional feature-space
    vocab  - The K x V matrix of vocabulary distributions.
    tau    - the row variance of A is tau^2 I_K
    sigma  - the variance in the estimation of the topic memberships. lambda ~ N(A'x, sigma^2I)
    '''
    
    tau   = 0.1
    sigma = 0.1
    
    Y     = rd.random((Q,P)).astype(DTYPE)
    omY   = np.identity(P, DTYPE)
    sigY  = np.identity(Q, DTYPE)
    
    U     = rd.random((K,Q)).astype(DTYPE)
    V     = rd.random((F,P)).astype(DTYPE)
    
    A     = U.dot(Y).dot(V.T)
    varA  = np.ones((F,1), DTYPE)
    
    # Vocab is K word distributions so normalize
    vocab = normalizerows_ip (rd.random((K, T)).astype(DTYPE))
    
    return VbSideTopicModelState(K, Q, F, P, T, A, varA, Y, omY, sigY, U, V, vocab, tau, sigma)

@autojit
def csr_indices(ptr, ind):
    '''
    Returns the indices of a CSR matrix, given its indptr and indices arrays.
    '''
    rowCount = len(ptr) - 1 
    
    rows = [0] * len(ind)
    totalElemCount = 0

    for r in range(rowCount):
        elemCount = ptr[r+1] - ptr[r]
        if elemCount > 0:
            rows[totalElemCount : totalElemCount + elemCount] = [r] * elemCount
        totalElemCount += elemCount

    return [rows, ind.tolist()]



def vec(A):
    return np.reshape(np.transpose(A), (-1,1))

def normalizerows_ip (matrix):
    '''
    Normalizes a matrix IN-PLACE.
    '''
    row_sums = matrix.sum(axis=1)
    matrix   /= row_sums[:, np.newaxis]
    return matrix

def rowwise_softmax (matrix):
    '''
    Assumes each row of the given matrix is an unnormalized distribution and
    uses the softmax metric to normalize it. This additionally uses some
    scaling to ensure that we never overflow.
    '''
    # TODO Just how compute intense is this method call?
    
    row_maxes = matrix.max(axis=1) # Underflow makes sense i.e. Pr(K=k) = 0. Overflow doesn't, i.e Pr(K=k) = \infty
    result    = np.exp(matrix - row_maxes[:, np.newaxis])
    result   /= result.sum(axis=1)[:,np.newaxis]
    return result

