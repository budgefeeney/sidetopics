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
import scipy.sparse.linalg as sla
import numpy.random as rd
import matplotlib.pyplot as plt

from util.overflow_safe import safe_log, safe_x_log_x, safe_log_one_plus_exp_of
from util.array_utils import normalizerows_ip, rowwise_softmax

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
    
    lmda - the DxK matrix of means of the topic distribution for each document
    nu   - the DxK the vector of variances of the topic distribution
    s    - The Dx1 vector of offsets.
    d    - the document index (for lambda and nu). If not specified we construct
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


def train(modelState, X, W, iterations=10000, epsilon=0.001, logInterval = 0):
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
       
    # Get ready to plot the evolution of the likelihood
    if logInterval > 0:
        elbos = np.zeros((iterations / logInterval,))
        iters = np.zeros((iterations / logInterval,))
    
    # We'll need the total word count per doc, and total count of docs
    docLen = W.sum(axis=1)
    D      = len(docLen)
    
    # No need to recompute this every time
    XTX = X.T.dot(X)
    
    # Identity matrices that occur
    I_PQ = np.ones((P,Q), DTYPE)
    I_P  = np.ones((P,P), DTYPE)
    I_Q  = np.ones((Q,Q), DTYPE)
    I_F  = np.ones((F,F), DTYPE)
    
    # Assign initial values to the query parameters
    lmda = rd.random((D, K)).astype(DTYPE)
    nu   = np.ones((D, K), DTYPE)
    s    = np.zeros((D,), DTYPE)
    lxi  = negJakkola (np.ones((D,K), DTYPE))
    
    # If we don't bother optimising either tau or sigma we can just do all this here once only 
    tsq     = tau * tau;
    ssq     = sigma * sigma;
    overTsq = 1. / tsq
    overSsq = 1. / ssq
    over2Ssq = 0.5 * overSsq;
    over2Tsq = 0.5 * overTsq;
    
    varA = 1./K * sla.inv (overTsq * I_F + overSsq * XTX)
   
    for iteration in range(iterations):
        
        # Save repeated computation
#         tsq     = tau * tau;
#         ssq     = sigma * sigma;
#         overTsq = 1. / tsq
#         overSsq = 1. / ssq
#         over2Ssq = 0.5 * overSsq;
#         over2Tsq = 0.5 * overTsq;
        
        # =============================================================
        # E-Step
        #   Model dists are q(Theta|A;Lambda;nu) q(A|Y) q(Y) and q(Z)....
        #   Where lambda is the posterior mean of theta.
        # =============================================================
        
        # Y, sigY, omY
        # 
        VTV = V.T.dot(V)
        UTU = U.T.dot(U)
        
        y = la.inv(I_PQ + np.kron (VTV, UTU)).dot(vec(U.dot(A).dot(V.T)))
        Y = np.reshape(y, (P,Q))
        sigY = 1./P * la.inv(np.trace(omY)  * I_Q + overTsq * np.trace(omY.dot(VTV) * UTU))
        omY  = 1./Q * la.inv(np.trace(sigY) * I_P + overTsq * np.trace(sigY.dot(UTU)) * VTV)
        
        _quickPrintElbo ("E-Step: q(Y)", iteration, X, W, K, Q, F, P, T, A, omA, Y, omY, sigY, U, V, vocab, tau, sigma, lmda, nu, lxi, s, docLen)
        
        # A, varA
        #
        # TODO, since only tau2sig2 changes at each step, would it be possible just to
        # amend the old inverse?
        # TODO Use sparse inverse
        A = K * varA.dot(overTsq * U.dot(Y).dot(V.T) + overSsq * lmda.T.dot(X))
        _quickPrintElbo ("E-Step: q(Y)", iteration, X, W, K, Q, F, P, T, A, omA, Y, omY, sigY, U, V, vocab, tau, sigma, lmda, nu, lxi, s, docLen)
       
        #
        # lmda_dk
        XAT = X.dot(A.T)
        lnVocab = safe_log (vocab)
        Z   = rowwise_softmax (lmda[:,:,np.newaxis] + lnVocab[np.newaxis,:,:]) # Z is DxKxT
        rho = 2 * s[:,np.newaxis] * lxi - 0.5 \
            + np.einsum('dt,dkt->dk', W, Z) / docLen[:,np.newaxis]
        
        rhs  = docLen[:,np.newaxis] * rho + overSsq * XAT
        lmda = rhs / (docLen[:,np.newaxis] * 2 * lxi + overSsq)
        
        _quickPrintElbo ("E-Step: q(Y)", iteration, X, W, K, Q, F, P, T, A, omA, Y, omY, sigY, U, V, vocab, tau, sigma, lmda, nu, lxi, s, docLen)
             
        
        #
        # nu_dk
        # TODO Double check this again...
        nu = 1./ np.sqrt(2. * docLen[:, np.newaxis] * lxi + overSsq)
        _quickPrintElbo ("E-Step: q(Y)", iteration, X, W, K, Q, F, P, T, A, omA, Y, omY, sigY, U, V, vocab, tau, sigma, lmda, nu, lxi, s, docLen)
       
        #
        # s_d
        s = (K/4. - 0.5 + (lxi * lmda).sum(axis = 1)) / lxi.sum(axis=1)
        _quickPrintElbo ("E-Step: s_d", iteration, X, W, K, Q, F, P, T, A, omA, Y, omY, sigY, U, V, vocab, tau, sigma, lmda, nu, lxi, s, docLen)

        #
        # xi_dk
        lxi = negJakkolaOfDerivedXi(lmda, nu, s)
        _quickPrintElbo ("E-Step: \u039B(xi_dk)", iteration, X, W, K, Q, F, P, T, A, omA, Y, omY, sigY, U, V, vocab, tau, sigma, lmda, nu, lxi, s, docLen)
       
       
        # =============================================================
        # M-Step
        #    Parameters for the softmax bound: lxi and s <-- ?
        #    The projection used for A: U and V
        #    The vocabulary : vocab
        #    The variances: tau, sigma
        # =============================================================
        
               
        # U
        #
        U = A.dot(V).dot(Y.T).dot (la.inv(Y.dot(V).dot(V.T).dot(Y.T) + np.trace(omY.dot(V).dot(V.T)) * sigY))
        _quickPrintElbo ("E-Step: q(Y)", iteration, X, W, K, Q, F, P, T, A, omA, Y, omY, sigY, U, V, vocab, tau, sigma, lmda, nu, lxi, s, docLen)

        # V
        # 
        V = A.T.dot(U).dot(Y).dot (la.inv(Y.T.dot(U.T).dot(U).dot(Y) + np.trace(sigY.dot(U.T).dot(U)) * omY))
        
        #
        # vocab
        #
        # TODO, since vocab is in the RHS, is there any way to optimize this?
        Z = rowwise_softmax (lmda[:,:,np.newaxis] + lnVocab[np.newaxis,:,:]) # Z is DxKxV
        vocab = normalizerows_ip (np.einsum('dt,dkt->kt', W, Z))
        _quickPrintElbo ("E-Step: q(Y)", iteration, X, W, K, Q, F, P, T, A, omA, Y, omY, sigY, U, V, vocab, tau, sigma, lmda, nu, lxi, s, docLen)

        
        
        if (logInterval > 0) and (iteration % logInterval == 0):
            elbo = varBound ( \
                VbSideTopicModelState (K, Q, F, P, T, A, varA, Y, omY, sigY, U, V, vocab, tau, sigma), \
                VbSideTopicQueryState(lmda, nu, lxi, s, docLen),
                X, W, Z, lnVocab, XAT, XTX)
                
            elbos[iteration / logInterval] = elbo
            iters[iteration / logInterval] = iteration
            print ("Iteration %5d  ELBO %f" % (iteration, elbo))
        
    if logInterval > 0:
        plot_bound(iters, elbos)
    
    return (VbSideTopicModelState (K, Q, F, P, T, A, varA, Y, omY, sigY, U, V, vocab, tau, sigma), \
            VbSideTopicQueryState (lmda, nu, lxi, s, docLen))
    
def plot_bound (iters, bounds):
    '''
    Plots the evoluation of the variational bound. The input is a pair of
    matched arrays: for a given point i, iters[i] was the iteration at which
    the bound bounds[i] was calculated
    '''
    
    fig  = plt.figure()
    plot = fig.add_subplot(1,1,1)
    plot.plot (iters, bounds)
    plot.set_ylabel("Bound")
    plot.set_xlabel("Iteration")
    plot.set_xticks(np.arange(0, len(bounds), max(1, len(bounds) / MAX_X_TICKS_PER_PLOT)))
    plt.show()
    
def _quickPrintElbo (updateMsg, iteration, X, W, K, Q, F, P, T, A, varA, Y, omY, sigY, U, V, vocab, tau, sigma, lmda, nu, lxi, s, docLen):
    '''
    Calculates the variational lower bound and prints it to stdout,
    prefixed with a tabl and the given updateMsg
    
    See varBound() for a full description of all parameters
    
    Obviously this is a very ugly inefficient method.
    '''
#     if iteration % 100 != 0:
#         return
    
    xi = deriveXi(lmda, nu, s)
    elbo = varBound ( \
                      VbSideTopicModelState (K, Q, F, P, T, A, varA, Y, omY, sigY, U, V, vocab, tau, sigma), \
                      VbSideTopicQueryState(lmda, nu, lxi, s, docLen), \
                      X, W)
    
    
    print ("\t Update %-30s  ELBO : %12.3f  lmda.mean=%f \tnu.mean=%f \txi.mean=%f \ts.mean=%f" % (updateMsg, elbo, lmda.mean(), nu.mean(), xi.mean(), s.mean()))

def varBound (modelState, queryState, X, W, Z = None, lnVocab = None, XAT=None, XTX = None):
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
    lnVocab    - the KxV matrix of the natural log applied to the vocabulary. Recalculated if
                 not provided
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
    
    if lnVocab is None:
        lnVocab = safe_log(vocab)
    if Z is None:
        Z = rowwise_softmax (lmda[:,:,np.newaxis] + lnVocab[np.newaxis,:,:]) # Z is DxKxV
    
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
    halfKF = 0.5 * K * F
    halfTsq = 0.5 / (tau * tau)
    lnP_A = -halfKF * LOG_2PI - halfKF * log (tau * tau) \
            -halfTsq * (np.sum(sigY * V.T.dot(V)) * np.sum(omY * U.T.dot(U)) \
                      + np.trace(omA.dot(XTX)) * K \
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
    
    lnP_Z = 0.0
    lnP_Z -= np.sum(docLenLmdaLxi * lmda)
    lnP_Z -= np.sum(docLen[:, np.newaxis] * nu * nu * lxi)
    lnP_Z += 2 * np.sum (s[:, np.newaxis] * docLenLmdaLxi)
    lnP_Z -= 0.5 * np.sum (docLen[:, np.newaxis] * lmda)
    lnP_Z += np.sum (lmda * np.sum(Z, axis=1)) # <-- Need to add simple sum over Z here
    lnP_Z -= np.sum(docLen[:,np.newaxis] * lxi * ((s**2)[:,np.newaxis] - xi**2))
    lnP_Z += 0.5 * np.sum(docLen[:,np.newaxis] * (s[:,np.newaxis] + xi))
    lnP_Z -= np.sum(docLen[:,np.newaxis] * safe_log_one_plus_exp_of(xi))
    lnP_Z -= np.sum (docLen * s)
        
    # <ln p(W|Z, vocab)>
    # 
    lnP_W = np.sum(lnVocab * np.einsum('dt,dkt->kt', W, Z))   # <-- Part of p(W)
    
    # ent1 is H[q(Theta)]
    ent_Theta = 0.5 * (K * LOG_2PI_E + np.sum (np.log(nu * nu)))
    
    # ent2 is H[q(A|Y)]
    ent_A = 0.5 * (F * K * LOG_2PI_E + K * log (la.det(omA)) + F * K * log (tau2))
    
    # ent3 is H[q(Y)]
    ent_Y = 0.5 * (P * K * LOG_2PI_E + Q * log (la.det(omY)) + P * log (la.det(sigY)))
    
    result = lnP_Y + lnP_A + +lnP_Theta + lnP_Z + ent_Y + ent_A + ent_Theta
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
    
    A     = U.dot(Y).dot(V)
    varA  = np.ones((F,1), DTYPE)
    
    # Vocab is K word distributions so normalize
    vocab = normalizerows_ip (rd.random((K, T)).astype(DTYPE))
    
    return VbSideTopicModelState(K, Q, F, P, T, A, varA, Y, omY, sigY, U, V, vocab, tau, sigma)

def vec(A):
    np.reshape(np.transpose(A), (-1,1))

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

