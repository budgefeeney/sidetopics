#!/usr/bin/python
# -*- coding: utf-8 -*- 

'''
The inputs and outputs of a SideTopicModel

Created on 29 Jun 2013

@author: bryanfeeney
'''

from collections import namedtuple
import numpy as np
import scipy.linalg as la
import numpy.random as rd
import scipy as sp
import scipy.sparse as ssp
import sys

# TODO Consider using numba for autojit (And jit with local types)
# TODO Investigate numba structs as an alternative to namedtuples
# TODO Make random() stuff predictable, either by incorporating a RandomState instance into model parameters
#      or calling a global rd.seed(0xC0FFEE) call.
# TODO The solutions for A V U etc all look similar to ridge regression - could
#      they be replaced with calls to built-in solvers?
# TODO Sigma and Tau optimisation is hugely expensive, not only because of their own updates,
#      but because were they fixed, we wouldn't need to do any updates for varA, which would save 
#      us from doing a FxF inverse at every iteration. 
# TODO varA is a huge, likely dense, FxF matrix
# TODO varV is a big, dense, PxP matrix...

VbSideTopicQueryState = namedtuple ( \
    'VbSideTopicState', \
    'lmda nu lxi s'\
)


def negJakkola(vec):
    '''
    The negated version of the Jakkola expression which was used in Bourchard's NIPS
    2007 softmax bound
    '''
    
    # COPY AND PASTE BETWEEN THIS AND negJakkolaOfDerivedXi()
    return 1./(2*vec) * (1./(1 - np.exp(-vec)) - 0.5)

def negJakkolaOfDerivedXi(lmda, nu, s, d = None):
    '''
    The negated version of the Jakkola expression which was used in Bourchard's NIPS
    2007 softmax bound calculating using the estimate of xi using lambda, nu, and s
    
    lmda - the DxK matrix of means of the topic distribution for each document
    nu   - the DxK the vector of variances of the topic distribution
    s    - The Dx1 vector of offsets.
    d    - the document index (for lambda and nu). If not specified we construct
           the full matrix of A(xi_dk)
    '''
    
    # COPY AND PASTE BETWEEN THIS AND negJakkola()
    if d is not None:
        vec = (np.sqrt (lmda[d,:] ** 2 -2 *lmda[d,:] * s[d] + s[d]**2 + nu[d,:]**2))
        return 1./(2*vec) * (1./(1 - np.exp(-vec)) - 0.5)
    else:
        mat = np.sqrt(lmda ** 2 - 2 * lmda * s[:, np.newaxis] + (s**2)[:, np.newaxis] + nu**2)
        return 1./(2 * mat) * (1./(1 - np.exp(-mat)) - 0.5)

def train(modelState, X, W, iterations=1000, epsilon=0.001):
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
    
    s      - A D-dimensional vector describing offset in our bound on the true value of ln sum_k e^theta_dk 
    lxi    - A DxK matrix used in the above bound, containing the negative Jakkola function applied to the 
             quadratic term xi
    lambda - the topics we've inferred for the current batch of documents
    nu     - the variance of topics we've inferred (independent)
    '''
    # Unpack the model state tuple for ease of use and maybe speed improvements
    (K, F, T, P, A, varA, V, varV, U, sigma, tau, vocab) = (modelState.K, modelState.F, modelState.V, modelState.P, modelState.A, modelState.varA, modelState.V, modelState.varV, modelState.U, modelState.sigma, modelState.tau, modelState.vocab)
       
    # We'll need the total word count per document
    docLen = W.sum(axis=1)
    
    # No need to recompute this every time
    XTX = X.T.dot(X)
    
    # Assign initial values to the query parameters
    D    = np.size(W, 0)
    lmda = rd.random((D, K))
    nu   = np.ones((D,K), np.float32)
    s    = np.zeros((D,))
    lxi  = negJakkola (np.ones((D, K), np.float32))
    
    XA = X.dot(A)
    for iteration in xrange(iterations):
        # Poor man's logging
        print("Iteration %d" % iteration)
        
        # Save repeated computation
        tsq      = tau * tau;
        tsqIP    = tsq * np.eye(P)
        trTsqIK  = K * tsq
        halfSig2 = 1./(sigma*sigma)
        tau2sig2 = (tau * tau) / (sigma * sigma)
        
        #
        # Inference Step 1: Update local parameters given model parameters
        #
        
        #
        # lmda_dk. rho is DxK
        pred = halfSig2 * XA
        Z    = rowwise_softmax(lmda)
        like = (W.dot(vocab.T) * Z) / docLen[:, np.newaxis]
        rho  = 2 * s[:,np.newaxis] * lxi -0.5 - like # TODO Verify this against the code
        
        lmda = 1. / (2 * docLen[:, np.newaxis] * lxi + 1./0.1**2)  \
             * rho * docLen[:, np.newaxis] - pred
        
        #
        # nu_dk
        nu = 2 * docLen[:, np.newaxis] * lxi + halfSig2
        
        #
        # xi_dk
        lxi = negJakkolaOfDerivedXi(lmda, nu, s)
        
        #
        # s_d
        s = (lxi * lmda + K/4.).sum(axis = 1) / lxi.sum(axis=1)
        
        #
        # Inference Step 2: Update model parameters given local parameters
        # 
        
        #
        # vocab
        #
        #     z_dvk  = 1/S phi_kv * exp(lmda[d,k])
        #
        #     phi_kv = 1/Z sum_d w_dv * z_dvk
        #            = 1/Z sum_d w_dv * 1/S phi_kv * exp(lmda[d,k])
        #            = 1/Z 1/S sum_d w_dv * phi_kv * exp(lmda[d,k])
        #            = phi_kv * 1/Z 1/S sum_d w_dv * exp(lmda[d,k])
        #            = phi_kv * 1/Z sum_d w_dv * exp(lmda[d,k])       (as 1/S gets embedded in 1/Z)
        # 
        # 
        vocab *= normalizerows (np.exp(lmda).T.dot(W))
        
        #
        # V, varV
        varV = la.inv (tsqIP + U.T.dot(U))
        V    = varV.dot(U.T).dot(A)
        
        #
        # A, varA
        # TODO, since only tau2sig2 changes at each step, would it be possible just to
        # amend the old inverse?
        varA = la.inv (np.eye(F) + tau2sig2 * XTX)
        A    = varA.dot (U.dot(V) + X.T.dot(lmda))
        XA   = X.dot(A)
        
        #
        # U
        U = A.dot(V.T).dot (la.inv(trTsqIK * varV + V.dot(V.T)))
        
        #
        # sigma
        #    Equivalent to \frac{1}{DK} \left( \sum_d (\sum_k nu_{dk}) + tr(\Omega_A) x_d^{T} \Sigma_A x_d + (\lambda - A^{T} x_d)^{T}(\lambda - A^{T} x_d) \right)
        #
        sigma = 1./(D*K) * (np.sum(nu) + D*K * tsq * np.sum(XTX * varA) + np.sum((lmda - XA)**2))
        
        #
        # tau
        #    Equivalent to \frac{1}{KF} \left( tr(\Sigma_A)tr(\Omega_A) + tr(\Sigma_V U U^{T})tr(\Omega_V) + tr ((M_A - U M_V)^{T} (M_A - U M_V)) \right)
        #
        varA_U = varA.dot(U)
        tau1 = np.trace(varA)*K*tsq
        tau2 = sum(varA_U[p,:].dot(U[p,:]) for p in xrange(P)) * K * tsq
        tau3 = np.sum((A - U.dot(V)) ** 2)
        
        tau = 1./(K*F) * (tau1 + tau2 + tau3)
        
        
    return (modelState, VbSideTopicQueryState(lmda, nu, lxi, s))



VbSideTopicModelState = namedtuple ( \
    'VbSideTopicState', \
    'K F T P A varA V varV U sigma tau vocab'\
)

def newVbModelState(K, F, T, P):
    '''
    Creates a new model state object for a topic model based on side-information. This state
    contains all parameters that once trained can be kept fixed for querying.
    
    The parameters are
    
    K - the number of topics
    F - the number of features
    P - the number of features in the projected space, P << F
    T - the number of terms in the vocabulary
    
    The returned object will contain K, F, V and P and also
    
    A      - the mean of the F x K matrix mapping F features to K topics
    varA   - the column variance of the distribution over A
    tau    - the row variance of A is tau^2 I_K
    V      - the mean of the P x K matrix mapping P projected features to K topics
    varV   - the column variance of the distribution over V (the row variance is again
             tau^2 I_K
    U      - the F x P projection matrix, such that A = UV
    sigma  - the variance in the estimation of the topic memberships lambda ~ N(A'x, sigma^2I)
    vocab  - The K x V matrix of voabularly distributions.
    '''
    
    V     = rd.random((P, K))
    varV  = np.identity(P, np.float32)
    U     = rd.random((F, P))
    A     = np.dot(U, V)
    varA  = np.identity(F, np.float32)
    tau   = 0.1
    sigma = 0.1
    
    # Vocab is K word distributions so normalize
    vocab = normalizerows (rd.random((K, T)))
    
    return VbSideTopicModelState(K, F, T, P, A, varA, V, varV, U, sigma, tau, vocab)

def normalizerows (matrix):
    row_sums = matrix.sum(axis=1)
    matrix   /= row_sums[:, np.newaxis]
    return matrix

def rowwise_softmax (matrix):
    '''
    Assumes each row of the given matrix is an unnormalized distribution and
    uses the softmax metric to normalize it. This additionally uses some
    scaling to ensure that we never underflow.
    '''
    # TODO Just how compute intense is this method call?
    
    row_maxes = matrix.max(axis=1) # Underflow makes sense i.e. Pr(K=k) = 0. Overflow doesn't, i.e Pr(K=k) = \infty
    matrix  -= row_maxes[:, np.newaxis]
    matrix   = np.exp(matrix)
    row_sums = matrix.sum(axis=1)
    matrix   /= row_sums[:, np.newaxis]
    return matrix

