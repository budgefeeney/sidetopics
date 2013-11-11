#!/usr/bin/python
# -*- coding: utf-8 -*- 

'''
This implements the side-topic model where

 * Topics are defined as a function of a side-information vector x and
   a matrix A
 * A is in turn defined as the product U Y V' with Y having a zero mean
   normal distribution
 * The distribution of Y is a multivariate distribution (i.e it's the
   distribution of vec(Y)). This means it has just one covariance matrix,
   instead of separate row and column covariances. 

Created on 29 Jun 2013

@author: bryanfeeney
'''

from math import log
from math import e
import numpy as np
import scipy.linalg as la
import scipy.sparse as ssp
import numpy.random as rd

from util.overflow_safe import safe_log, safe_log_one_plus_exp_of
from util.array_utils import normalizerows_ip
from model.sidetopic_uyv import DTYPE, LOG_2PI, LOG_2PI_E, _quickPrintElbo, \
    VbSideTopicModelState,  VbSideTopicQueryState, \
    log_likelihood, plot_bound, query, negJakkola, deriveXi, \
    sparseScalarProductOfDot, sparseScalarQuotientOfDot
from util.vectrans import vec, vec_transpose, vec_transpose_csr, sp_vec_trans_matrix

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
# CODE
# ==============================================================



def train(modelState, X, W, iterations=10000, epsilon=0.001, logInterval = 0, plotInterval = 0, fastButInaccurate=False):
    '''
    Creates a new query state object for a topic model based on side-information. 
    This contains all those estimated parameters that are specific to the actual
    date being queried - this must be used in conjunction with a model state.
    
    The parameters are
    
    modelState - the model state with all the model parameters
    X          - the D x F matrix of side information vectors
    W          - the D x V matrix of word **count** vectors.
    iterations - how long to iterate for
    epsilon    - currently ignored, in future, allows us to stop early.
    logInterval  - the interval between iterations where we calculate and display
                   the log-likelihood bound
    plotInterval - the interval between iterations we we display the log-likelihood
                   bound values calcuated at each log-interval
    fastButInaccurate - if true, we may use a psedo-inverse instead of an inverse
                        when solving for Y when the true inverse is unavailable.
    
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
    
    if W.dtype.kind == 'i':      # for the sparseScalorQuotientOfDot() method to work
        W = W.astype(np.float32)
    
    # Get ready to plot the evolution of the likelihood
    if logInterval > 0:
        elbos = np.zeros((iterations / logInterval,))
        likes = np.zeros((iterations / logInterval,))
        iters = np.zeros((iterations / logInterval,))
    iters.fill(-1)
    
    # We'll need the total word count per doc, and total count of docs
    docLen = np.squeeze(np.asarray (W.sum(axis=1))) # Force to a one-dimensional array for np.newaxis trick to work
    D      = len(docLen)
    
    # No need to recompute this every time
    XTX = X.T.dot(X)
    
    # Identity matrices that occur
    I_P  = np.eye(P,P,     0, DTYPE)
    I_Q  = np.eye(Q,Q,     0, DTYPE)
    I_QP = np.eye(Q*P,Q*P, 0, DTYPE)
    I_F  = ssp.eye(F,F,    0, DTYPE, "csc") # X is CSR, XTX is consequently CSC, sparse inverse requires CSC
    
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
        # If U'U is invertible, use inverse to convert Y to a Sylvester eqn
        # which has a much, much faster solver. Recall update for Y is of the form
        #   Y + AYB = C where A = U'U, B = V'V and C=U'AV
        # 
        VTV = V.T.dot(V)
        UTU = U.T.dot(U)
        
        sigy = la.inv(I_QP + overTsqSsq * np.kron(VTV, UTU))
        _quickPrintElbo ("E-Step: q(Y) [sigY]", iteration, X, W, K, Q, F, P, T, A, omA, Y, omY, sigY, U, V, vocab, tau, sigma, expLmda, nu, lxi, s, docLen)
        
        Y = overTsqSsq * sigy.dot(vec(U.T.dot(A).dot(V)))
        _quickPrintElbo ("E-Step: q(Y) [Mean]", iteration, X, W, K, Q, F, P, T, A, omA, Y, omY, sigY, U, V, vocab, tau, sigma, expLmda, nu, lxi, s, docLen)
        
        # A 
        #
        # So it's normally A = (UYV' + L'X) omA with omA = inv(t*I_F + s*XTX)
        #   so A inv(omA) = UYV' + L'X
        #   so inv(omA)' A' = VY'U' + X'L
        # at which point we can use a built-in solve
        #
#       A = (overTsq * U.dot(Y).dot(V.T) + X.T.dot(expLmda).T).dot(omA)
        lmda = np.log(expLmda, out=expLmda)
        A = la.solve(tI_sXTX, X.T.dot(lmda) + V.dot(Y.T).dot(U.T)).T
        np.exp(expLmda, out=expLmda)
        _quickPrintElbo ("E-Step: q(A)", iteration, X, W, K, Q, F, P, T, A, omA, Y, omY, sigY, U, V, vocab, tau, sigma, expLmda, nu, lxi, s, docLen)
       
        # lmda_dk, nu_dk, s_d, and xi_dk
        #
        XAT = X.dot(A.T)
        query (VbSideTopicModelState (K, Q, F, P, T, A, omA, Y, omY, sigY, U, V, vocab, tau, sigma), \
               X, W, \
               VbSideTopicQueryState(expLmda, nu, lxi, s, docLen), \
               scaledWordCounts=scaledWordCounts, \
               XAT = XAT, \
               iterations=10, \
               logInterval = 0, plotInterval = 0)
       
       
        # =============================================================
        # M-Step
        #    Parameters for the softmax bound: lxi and s
        #    The projection used for A: U and V
        #    The vocabulary : vocab
        #    The variances: tau, sigma
        # =============================================================
               
        # U
        # 
        U = A.dot(V).dot(Y.T).dot (la.inv( \
                Y.T.dot(U.T).dot(U).dot(Y) + 
        ))
        _quickPrintElbo ("M-Step: U", iteration, X, W, K, Q, F, P, T, A, omA, Y, omY, sigY, U, V, vocab, tau, sigma, expLmda, nu, lxi, s, docLen)

        # V
        # 
        V = A.T.dot(U).dot(Y).dot (la.inv(Y.T.dot(U.T).dot(U).dot(Y) + np.trace(sigY.dot(U.T).dot(U)) * omY))
        _quickPrintElbo ("M-Step: V", iteration, X, W, K, Q, F, P, T, A, omA, Y, omY, sigY, U, V, vocab, tau, sigma, expLmda, nu, lxi, s, docLen)

        # vocab
        #
        factor = (scaledWordCounts.T.dot(expLmda)).T # Gets materialized as a dense matrix...
        vocab *= factor
        normalizerows_ip(vocab)
        _quickPrintElbo ("M-Step: \u03A6", iteration, X, W, K, Q, F, P, T, A, omA, Y, omY, sigY, U, V, vocab, tau, sigma, expLmda, nu, lxi, s, docLen)
        
        # =============================================================
        # Handle logging of variational bound, likelihood, etc.
        # =============================================================
        if (logInterval > 0) and (iteration % logInterval == 0):
            modelState = VbSideTopicModelState (K, Q, F, P, T, A, omA, Y, omY, sigY, U, V, vocab, tau, sigma)
            queryState = VbSideTopicQueryState(expLmda, nu, lxi, s, docLen)
            
            elbo   = varBound (modelState, queryState, X, W, None, XAT, XTX)
            likely = log_likelihood(modelState, X, W, queryState) #recons_error(modelState, X, W, queryState)
                
            elbos[iteration / logInterval] = elbo
            iters[iteration / logInterval] = iteration
            likes[iteration / logInterval] = likely
            print ("Iteration %5d  ELBO %15f   Log-Likelihood %15f" % (iteration, elbo, likely))
        
        if (plotInterval > 0) and (iteration % plotInterval == 0) and (iteration > 0):
            plot_bound(iters, elbos, likes)
            
#        if (iteration % 10 == 0) and (iteration > 0):
#            print ("\n\nOmega_Y[0,:] = " + str(omY[0,:]))
#            print ("Sigma_Y[0,:] = " + str(sigY[0,:]))
            
    
    # Right before we end, plot the evoluation of the bound and likelihood
    # if we've been asked to do so.
    if plotInterval > 0:
        plot_bound(iters, elbos, likes)
    
    return VbSideTopicModelState (K, Q, F, P, T, A, omA, Y, omY, sigY, U, V, vocab, tau, sigma), \
           VbSideTopicQueryState (expLmda, nu, lxi, s, docLen)


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
    (expLmda, nu, lxi, s, docLen) = (queryState.expLmda, queryState.nu, queryState.lxi, queryState.s, queryState.docLen)
    
    lmda = np.log(expLmda)
    
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
    scaledWordCounts = sparseScalarQuotientOfDot(W, expLmda, vocab, out=scaledWordCounts)

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
    lnP_w_dt = sparseScalarProductOfDot(scaledWordCounts, expLmda, vocab * safe_log(vocab))
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
    
    return result


def newVbModelState(K, Q, F, P, T):
    '''
    Creates a new model state object for a topic model based on side-information. This state
    contains all parameters that o§nce trained can be kept fixed for querying.
    
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


