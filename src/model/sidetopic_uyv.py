#!/usr/bin/python
# -*- coding: utf-8 -*- 

'''
This implements the side-topic model where

 * Topics are defined as a function of a side-information vector x and
   a matrix A
 * A is in turn defined as the product U Y V' with Y having a zero mean
   normal distribution

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
DTYPE = np.float64

LOG_2PI   = log(2 * pi)
LOG_2PI_E = log(2 * pi * e)

# ==============================================================
# TUPLES
# ==============================================================

VbSideTopicTrainPlan = namedtuple ( \
    'VbSideTopicTrainPlan',
    'iterations', 'epsilon', 'logFrequency', 'plot', 'plotFile', 'plotIncremental', 'fastButInaccurate')                            

VbSideTopicQueryState = namedtuple ( \
    'VbSideTopicState', \
    'expLmda nu lxi s docLen'\
)


VbSideTopicModelState = namedtuple ( \
    'VbSideTopicState', \
    'K, Q, F, P, T, A, varA, Y, omY, sigY, sigT, U, V, vocab, topicVar, featVar, lowTopicVar, lowFeatVar'
)

# ==============================================================
# CODE
# ==============================================================

def newInferencePlan(iterations=100, epsilon=0.1, logFrequency=0, plot=True, plotFile=None, plotIncremental=False, fastButInaccurate=None):
    '''
    Creates an inference plan to be used by either train() or query(),
    which specifies how long to run the inference for, how often to 
    measure and log the bound, whether to create a plot of the bound,
    which file to save it to (it's displayed on-screen otherwise) and
    whether optimisations should be used that improve the speed of 
    inference at the cost of accuracy/
    
    iterations - how long to iterate for
    epsilon    - should the last measure of the bound and the current one differ
                 by less than this amount we assume the model has converged and
                 stop inference.
    logFrequency - how many times should we inspect the variational bound. This is
                   done on a power-scale, so for example for 256 iterations, if we set
                   this to 8, we'll measure at iterations 1, 2, 4, 16, 32, 64, 128 and 255.
    plotIncremental - create a plot every time we measure the bound
    plot     - should we show a plot of the measured bound after running or not
    plotFile - where to save the plot of the bound. If None the plot is shown
               on screen.
    fastButInaccurate - if true, optimisations that are theoretically unsound or
                        which degrade results may be used.
    '''
    return VbSideTopicTrainPlan(iterations, epsilon, logFrequency, plot, plotFile, plotIncremental, fastButInaccurate)

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
    
    lmda    - the DxK matrix of means of the topic distribution for each document.
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


def train(modelState, X, W, plan):
    '''
    Creates a new query state object for a topic model based on side-information. 
    This contains all those estimated parameters that are specific to the actual
    date being queried - this must be used in conjunction with a model state.
    
    The parameters are
    
    modelState - the model state with all the model parameters
    X          - the D x F matrix of side information vectors
    W          - the D x V matrix of word **count** vectors.
    
    
    This returns a tuple of new model-state and query-state. The latter object will
    contain X and W and also
    
    s      - A D-dimensional vector describing the offset in our bound on the true value of ln sum_k e^theta_dk 
    lxi    - A DxK matrix used in the above bound, containing the negative Jakkola function applied to the 
             quadratic term xi
    lambda - the topics we've inferred for the current batch of documents
    nu     - the variance of topics we've inferred (independent)
    '''
    # Unpack the model state tuple for ease of use and maybe speed improvements
    K, Q, F, P, T, A, varA, Y, omY, sigY, sigT, U, V, vocab, sigmaSq, alphaSq, kappaSq, tauSq = modelState.K, modelState.Q, modelState.F, modelState.P, modelState.T, modelState.A, modelState.varA, modelState.Y, modelState.omY, modelState.sigY, modelState.sigT, modelState.U, modelState.V, modelState.vocab, modelState.topicVar, modelState.featVar, modelState.lowTopicVar, modelState.lowFeatVar
    iterations, epsilon, logCount, plot, plotFile, plotIncremental, fastButInaccurate = plan.iterations, plan.epsilon, plan.logFrequency, plan.plot, plan.plotFile, plan.plotIncremental, plan.fastButInaccurate
    
    if W.dtype.kind == 'i':      # for the sparseScalorQuotientOfDot() method to work
        W = W.astype(DTYPE)
    
    # Get ready to plot the evolution of the likelihood, with multiplicative updates (e.g. 1, 2, 4, 8, 16, 32, ...)
    if logCount > 0:
        multiStepSize = np.power (iterations, 1. / logCount)
        logIter = 1
        elbos = []
        likes = []
        iters = []
    else:
        logIter = iterations + 1
    
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
    overTsq = 1. / tauSq
    overSsq = 1. / sigmaSq
    overAsq = 1. / alphaSq
    overKsq = 1. / kappaSq
    
    varRatio = (alphaSq * sigmaSq) / (tauSq * kappaSq)
    
    # TODO the inverse being almost always dense means that it might
    # be faster to convert to dense and use the normal solver, despite
    # the size constraints.
#    varA = 1./K * sla.inv (overTsq * I_F + overSsq * XTX)
    aI_XTX = (overAsq * I_F + XTX).todense(); 
    omA = la.inv (aI_XTX)
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
        
        sigY = la.inv(overTsq * overKsq * I_Q + overAsq * overSsq * UTU)
        _quickPrintElbo ("E-Step: q(Y) [sigY]", iteration, X, W, K, Q, F, P, T, A, omA, Y, omY, sigY, sigT, U, V, vocab, sigmaSq, alphaSq, kappaSq, tauSq, expLmda, nu, lxi, s, docLen)
        
        omY  = la.inv(overTsq * overKsq * I_P + overAsq * overSsq * VTV) 
        _quickPrintElbo ("E-Step: q(Y) [omY]", iteration, X, W, K, Q, F, P, T, A, omA, Y, omY, sigY, sigT, U, V, vocab, sigmaSq, alphaSq, kappaSq, tauSq, expLmda, nu, lxi, s, docLen)
        
        try:
            invUTU = la.inv(UTU)
            Y = la.solve_sylvester (varRatio * invUTU, VTV, invUTU.dot(U.T).dot(A).dot(V))   
        except np.linalg.linalg.LinAlgError as e: # U'U seems to rapidly become singular (before 5 iters)
            if fastButInaccurate:                 
                invUTU = la.pinvh(UTU) # Obviously unstable, inference stalls much earlier than the correct form
                Y = la.solve_sylvester (varRatio * invUTU, VTV, invUTU.dot(U.T).dot(A).dot(V))  
            else:
                Y = np.reshape (la.solve(varRatio * I_QP + np.kron(VTV, UTU), vec(U.T.dot(A).dot(V))), (Q,P), 'F')
                
        _quickPrintElbo ("E-Step: q(Y) [Mean]", iteration, X, W, K, Q, F, P, T, A, omA, Y, omY, sigY, sigT, U, V, vocab, sigmaSq, alphaSq, kappaSq, tauSq, expLmda, nu, lxi, s, docLen)
        
        
        # A 
        #
        # So it's normally A = (UYV' + L'X) omA with omA = inv(t*I_F + s*XTX)
        #   so A inv(omA) = UYV' + L'X
        #   so inv(omA)' A' = VY'U' + X'L
        # at which point we can use a built-in solve
        #
        lmda = np.log(expLmda, out=expLmda)
        try:
            A = la.solve(aI_XTX, X.T.dot(lmda) + overAsq * V.dot(Y.T).dot(U.T)).T
        except ValueError as e:
            print(str(e))
            print ("Hmm")
        np.exp(expLmda, out=expLmda)
        _quickPrintElbo ("E-Step: q(A)", iteration, X, W, K, Q, F, P, T, A, omA, Y, omY, sigY, sigT, U, V, vocab, sigmaSq, alphaSq, kappaSq, tauSq, expLmda, nu, lxi, s, docLen)
       
        # lmda_dk, nu_dk, s_d, and xi_dk
        #
        XAT = X.dot(A.T)
        query (VbSideTopicModelState (K, Q, F, P, T, A, omA, Y, omY, sigY, sigT, U, V, vocab, sigmaSq, alphaSq, kappaSq, tauSq), \
               X, W, \
               VbSideTopicQueryState(expLmda, nu, lxi, s, docLen), \
               scaledWordCounts=scaledWordCounts, \
               XAT = XAT, \
               iterations=1, \
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
        U = A.dot(V).dot(Y.T).dot (la.inv(Y.dot(V.T).dot(V).dot(Y.T) + np.trace(omY.dot(V.T).dot(V)) * sigY))
        _quickPrintElbo ("M-Step: U", iteration, X, W, K, Q, F, P, T, A, omA, Y, omY, sigY, sigT, U, V, vocab, sigmaSq, alphaSq, kappaSq, tauSq, expLmda, nu, lxi, s, docLen)

        # V
        # 
        V = A.T.dot(U).dot(Y).dot (la.inv(Y.T.dot(U.T).dot(U).dot(Y) + np.trace(sigY.dot(U.T).dot(U)) * omY))
        _quickPrintElbo ("M-Step: V", iteration, X, W, K, Q, F, P, T, A, omA, Y, omY, sigY, sigT, U, V, vocab, sigmaSq, alphaSq, kappaSq, tauSq, expLmda, nu, lxi, s, docLen)

        # vocab
        #
        factor = (scaledWordCounts.T.dot(expLmda)).T # Gets materialized as a dense matrix...
        vocab *= factor
        normalizerows_ip(vocab)
        _quickPrintElbo ("M-Step: \u03A6", iteration, X, W, K, Q, F, P, T, A, omA, Y, omY, sigY, sigT, U, V, vocab, sigmaSq, alphaSq, kappaSq, tauSq, expLmda, nu, lxi, s, docLen)
        
        # =============================================================
        # Handle logging of variational bound, likelihood, etc.
        # =============================================================
        if iteration == logIter:
            modelState = VbSideTopicModelState (K, Q, F, P, T, A, omA, Y, omY, sigY, sigT, U, V, vocab, sigmaSq, alphaSq, kappaSq, tauSq)
            queryState = VbSideTopicQueryState(expLmda, nu, lxi, s, docLen)
            
            elbo   = varBound (modelState, queryState, X, W, None, XAT, XTX)
            likely = log_likelihood(modelState, X, W, queryState) #recons_error(modelState, X, W, queryState)
                
            elbos.append (elbo)
            iters.append (iteration)
            likes.append (likely)
            print ("Iteration %5d  ELBO %15f   Log-Likelihood %15f" % (iteration, elbo, likely))
            
            logIter = min (np.ceil(logIter * multiStepSize), iterations - 1)
        
        if plot and plotIncremental:
            plot_bound(plotFile + "-iter-" + str(iteration), np.array(iters), np.array(elbos), np.array(likes))
            
    
    # Right before we end, plot the evoluation of the bound and likelihood
    # if we've been asked to do so.
    if plot:
        plot_bound(plotFile, iters, elbos, likes)
    
    return VbSideTopicModelState (K, Q, F, P, T, A, omA, Y, omY, sigY, sigT, U, V, vocab, sigmaSq, alphaSq, kappaSq, tauSq), \
           VbSideTopicQueryState (expLmda, nu, lxi, s, docLen)


def query(modelState, X, W, queryState = None, scaledWordCounts=None, XAT = None, iterations=10, epsilon=0.001, logInterval = 0, plotInterval = 0):
    '''
    Determines the most likely topic memberships for the given documents as
    described by their feature and word matrices X and W. All  elements of
    the model are kept fixed. The query state, if provied, will be mutated 
    in-place, so one should make a defensive copy if this behaviour is 
    undesirable.
    
    Parameters
    modelState   - the model used to assign topics to documents. This is kept fixed
    X            - the DxF matrix of feature-vectors associated with the documents
    W            - The DxT matrix of word-count vectors representing the documents
    queryState   - the query-state object, with initial topic assignments. The members
                   of this are directly mutatated.
    scaledWordCounts - a DxT matrix with the same number of non-zero entries as W.
                       This is overwritten.
    XAT          - the product of X.dot(modelState.A.T)
    iterations   - the number of iterations to execute
    epsilon      - ignored
    logInterval  - the interval between iterations where we calculate and display
                   the log-likelihood bound
    plotInterval - the interval between iterations we we display the log-likelihood
                   bound values calculated at each log-interval
                   
    Returns
      The original query state, with the mutated in-place matrices
    '''
    
    K, Q, F, P, T, A, varA, Y, omY, sigY, sigT, U, V, vocab, sigmaSq, alphaSq, kappaSq, tauSq = modelState.K, modelState.Q, modelState.F, modelState.P, modelState.T, modelState.A, modelState.varA, modelState.Y, modelState.omY, modelState.sigY, modelState.sigT, modelState.U, modelState.V, modelState.vocab, modelState.topicVar, modelState.featVar, modelState.lowTopicVar, modelState.lowFeatVar
    if queryState is None:
        queryState = newVbQueryState(W, K)
    expLmda, nu, lxi, s, docLen = queryState.expLmda, queryState.nu, queryState.lxi, queryState.s, queryState.docLen
    
    overTsq, overSsq, overAsq, overKsq = 1./tauSq, 1./sigmaSq, 1./alphaSq, 1./kappaSq
    
    if W.dtype.kind == 'i':      # for the sparseScalorQuotientOfDot() method to work
        W = W.astype(DTYPE)
    if scaledWordCounts is None:
        scaledWordCounts = W.copy()
    if XAT is None:
        XAT = X.dot(A.T)
    
    for iteration in range(iterations):
        # sc = W / lmda.dot(vocab)
        scaledWordCounts = sparseScalarQuotientOfDot(W, expLmda, vocab, out=scaledWordCounts)
        
#        expLmdaCopy = expLmda.copy()
        
        rho = 2 * s[:,np.newaxis] * lxi - 0.5 \
            + expLmda * (scaledWordCounts.dot(vocab.T)) / docLen[:,np.newaxis]  
        rhs  = docLen[:,np.newaxis] * rho + overSsq * XAT
        
        expLmda[:] = rhs / (docLen[:,np.newaxis] * 2 * lxi + overSsq)
        # Note we haven't applied np.exp() yet, we're holding off till we've evaluated the next few terms
        # This efficiency saving only actually applies once we've disabled all the _quickPrintElbo calls
        
        _quickPrintElbo ("E-Step: q(\u03F4) [Mean]", iteration, X, W, K, Q, F, P, T, A, varA, Y, omY, sigY, sigT, U, V, vocab, sigmaSq, alphaSq, kappaSq, tauSq, np.exp(expLmda), nu, lxi, s, docLen)
         
        # nu_dk
        #
        nu[:] = 1./ np.sqrt(2. * docLen[:, np.newaxis] * lxi + overSsq)
        _quickPrintElbo ("E-Step: q(\u03F4) [Var] ", iteration, X, W, K, Q, F, P, T, A, varA, Y, omY, sigY, sigT, U, V, vocab, sigmaSq, alphaSq, kappaSq, tauSq, np.exp(expLmda), nu, lxi, s, docLen)
          
        # s_d
        #
        s[:] = (K/4. - 0.5 + (lxi * expLmda).sum(axis = 1)) / lxi.sum(axis=1)
        _quickPrintElbo ("E-Step: s_d", iteration, X, W, K, Q, F, P, T, A, varA, Y, omY, sigY, sigT, U, V, vocab, sigmaSq, alphaSq, kappaSq, tauSq, np.exp(expLmda), nu, lxi, s, docLen)
        
        # xi_dk
        # 
        lxi[:] = negJakkolaOfDerivedXi(expLmda, nu, s)
        _quickPrintElbo ("E-Step: \u039B(xi_dk)", iteration, X, W, K, Q, F, P, T, A, varA, Y, omY, sigY, sigT, U, V, vocab, sigmaSq, alphaSq, kappaSq, tauSq, np.exp(expLmda), nu, lxi, s, docLen)

        # Now finally we finish off the estimate of exp(lmda)
        np.exp(expLmda, out=expLmda)
        
    return VbSideTopicQueryState(expLmda, nu, lxi, s, docLen)
    

def _non_real_indices(A):
    indices = []
    for r in range(A.shape[0]):
        for c in range(A.shape[1]):
            if np.isinf(A[r,c]) or np.isnan(A[r,c]):
                indices.append((r,c))
    return indices

def plot_bound (plotFile, iters, bounds, likes):
    '''
    Plots the evolution of the variational bound and the log-likelihood. 
    The input is a pair of  matched arrays: for a given point i, iters[i]
    was the iteration at which the bound bounds[i] was calculated
    '''
    itersFilled = [i for i in iters if i >= 0]
    numValues   = len(itersFilled)
    boundsFilled = bounds if numValues == len(iters) else bounds[:numValues]
    likesFilled  = likes  if numValues == len(iters) else likes[:numValues]
    
    _, plot1  = plt.subplots()
    
    plot1.plot (itersFilled, boundsFilled, 'b-')
    plot1.set_ylabel("Bound", color='b')
    
    plot1.set_xlabel("Iteration")
#    xticks = np.arange(MAX_X_TICKS_PER_PLOT) if numValues < MAX_X_TICKS_PER_PLOT else np.arange(0, itersFilled.max(), itersFilled.max() / MAX_X_TICKS_PER_PLOT)
#    plot1.set_xticks(xticks)
    
    plot2 = plot1.twinx() # superimposed plot with the same x-axis (provides for two different y-axes)
    plot2.plot (itersFilled, likesFilled, 'g-')
    plot2.set_ylabel("Log Likelihood", color='g')
    
    if plotFile is None:
        plt.show()
    else:
        plt.savefig(plotFile + '.png')
    
def _quickPrintElbo (updateMsg, iteration, X, W, K, Q, F, P, T, A, varA, Y, omY, sigY, sigT, U, V, vocab, sigmaSq, alphaSq, kappaSq, tauSq, expLmda, nu, lxi, s, docLen):
    '''
    This checks that none of the matrix parameters contain a NaN or an Inf
    value, then calcuates the variational bound, and prints it to stdout with
    the given update message.
    
    A tremendously inefficient method for debugging only.
    '''
    def _nan (varName):
        print (str(varName) + " has NaNs")
    def _inf (varName):
        print (str(varName) + " has infs")
    
    
    # NaN tests
    if np.isnan(Y).any():
        _nan("Y")
    if omY is not None and np.isnan(omY).any():
        _nan("omY")
    if sigY is not None and np.isnan(sigY).any():
        _nan("sigY")
        
    if np.isnan(A).any():
        _nan("A")
    if np.isnan(varA).any():
        _nan("varA")
        
    if np.isnan(expLmda).any():
        _nan("expLmda")
    if np.isnan(sigT).any():
        _nan("sigT")
    if np.isnan(nu).any():
        _nan("nu")
        
    if U is not None and np.isnan(U).any():
        _nan("U")
    if V is not None and np.isnan(V).any():
        _nan("V")
        
    if np.isnan(vocab).any():
        _nan("vocab")
        
    # Infs tests
    if np.isinf(Y).any():
        _inf("Y")
    if omY is not None and np.isinf(omY).any():
        _inf("omY")
    if sigY is not None and np.isinf(sigY).any():
        _inf("sigY")
        
    if np.isinf(A).any():
        _inf("A")
    if np.isinf(varA).any():
        _inf("varA")
        
    if np.isinf(expLmda).any():
        _inf("expLmda")
    if sigY is not None and np.isinf(sigT).any():
        _inf("sigT")
    if np.isinf(nu).any():
        _inf("nu")
        
    if U is not None and np.isinf(U).any():
        _inf("U")
    if V is not None and np.isinf(V).any():
        _inf("V")
        
    if np.isinf(vocab).any():
        _inf("vocab")
    
    lmda = np.log(expLmda)
    xi = deriveXi(lmda, nu, s)
    elbo = varBound ( \
                      VbSideTopicModelState (K, Q, F, P, T, A, varA, Y, omY, sigY, sigT, U, V, vocab, sigmaSq, alphaSq, kappaSq, tauSq), \
                      VbSideTopicQueryState(expLmda, nu, lxi, s, docLen), \
                      X, W)
    
    
    print ("\t Update %5d: %-30s  ELBO : %12.3f  lmda.mean=%f \tlmda.max=%f \tlmda.min=%f \tnu.mean=%f \txi.mean=%f \ts.mean=%f" % (iteration, updateMsg, elbo, lmda.mean(), lmda.max(), lmda.min(), nu.mean(), xi.mean(), s.mean()))

def varBound (modelState, queryState, X, W, lnVocab = None, XAT=None, XTX = None, scaledWordCounts = None, UTU = None, VTV = None):
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
    UTU        - as above for U
    VTV        - as above for V
    
    Returns
        The (positive) variational lower bound
    '''
    
    # Unpack the model and query state tuples for ease of use and maybe speed improvements
    K, Q, F, P, T, A, varA, Y, omY, sigY, sigT, U, V, vocab, sigmaSq, alphaSq, kappaSq, tauSq = modelState.K, modelState.Q, modelState.F, modelState.P, modelState.T, modelState.A, modelState.varA, modelState.Y, modelState.omY, modelState.sigY, modelState.sigT, modelState.U, modelState.V, modelState.vocab, modelState.topicVar, modelState.featVar, modelState.lowTopicVar, modelState.lowFeatVar
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
    
    # If not already provided, we'll also need the following products
    #
    if XAT is None:
        XAT = X.dot(A.T)
    if XTX is None:
        XTX = X.T.dot(X)
    if V is not None and VTV is None:
        VTV = V.T.dot(V)
    if U is not None and UTU is None:
        UTU = U.T.dot(U)
        
    # also need one over the usual variances
    overSsq, overAsq, overKsq, overTsq = 1./sigmaSq, 1./alphaSq, 1./kappaSq, 1./tauSq
    overTkSq = overTsq * overKsq
    overAsSq = overAsq * overSsq
    
   
    # <ln p(Y)>
    #
    trSigY = 0 if sigY is None else np.trace(sigY)
    trOmY  = 0 if omY  is None else np.trace(omY)
    lnP_Y = -0.5 * (Q*P * LOG_2PI + overTkSq * trSigY * trOmY + overTkSq * np.sum(Y * Y))
    
    # <ln P(A|Y)>
    # TODO it looks like I should take the trace of omA \otimes I_K here.
    # TODO Need to check re-arranging sigY and omY is sensible.
    halfKF = 0.5 * K * F
    
    # Horrible, but varBound can be called by two implementations, one with Y as a matrix-variate
    # where sigY is QxQ and one with Y as a multi-varate, where sigY is a QPxQP.
    A_from_Y = Y.dot(U.T) if V is None else U.dot(A).dot(V.T)
    varFactorU = np.trace(sigY.dot(np.kron(VTV, UTU))) if sigY.shape[0] == Q*P else np.sum(sigY*UTU)
    varFactorV = 1 if V is None \
        else np.sum(omY * V.T.dot(V))
    lnP_A = -halfKF * LOG_2PI - halfKF * log (alphaSq) -halfKF * log(sigmaSq) \
            -0.5 * (overAsSq * varFactorV * varFactorU \
                      + np.trace(XTX.dot(varA)) * K \
                      + np.sum (np.square(A - A_from_Y)))
            
    # <ln p(Theta|A,X)
    # 
    lnP_Theta = -0.5 * D * LOG_2PI -0.5 * D * K * log (sigmaSq) \
                - 0.5 / sigmaSq * ( \
                    np.sum(nu) + D*K * np.sum(XTX * varA) + np.sum(np.square(lmda - XAT)))
    
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
    lnDetOmY  = 0 if omY  is None else log(la.det(omY))
    lnDetSigY = 0 if sigY is None else log(la.det(sigY))
    ent_Y = 0.5 * (P * K * LOG_2PI_E + Q * lnDetOmY + P * lnDetSigY)
    
    # H[q(A|Y)]
    #
    # A few things - omA is fixed so long as tau an sigma are, so there's no benefit in
    # recalculating this every time.
    #
    # However in a recent test, la.det(omA) = 0
    # this is very strange as omA is the inverse of (s*I + t*XTX)
    #
#    ent_A = 0.5 * (F * K * LOG_2PI_E + K * log (la.det(omA)) + F * K * log (tau2))\
    ent_A = 0
    
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

 
def recons_error (modelState, X, W, queryState):
    tpcs_inf = rowwise_softmax(queryState.lmda)
    W_inf    = np.array(tpcs_inf.dot(modelState.vocab) * queryState.docLen[:,np.newaxis], dtype=np.int32)
    return np.sum(np.square(W - W_inf)) / X.shape[0]


def log_likelihood(modelState, X, W, queryState):
    '''
    Returns the log likelihood of the given features and words according to the
    given model.
    
    modelState - the model, provided by #train() - to use to evaluate the data
    X          - the DxF matrix of features
    W          - the DxT matrix of words
    
    Return:
        The marginal likelihood of the data
    '''
    if W.dtype.kind == 'i':      # for the sparseScalorProductOf() method to work
        W = W.astype(DTYPE)
    
    F, T, vocab = modelState.F, modelState.T, modelState.vocab
    assert X.shape[1] == F, "Model is trained to expect " + str(F) + " features but feature-matrix has " + str(X.shape[1]) + " features"
    assert W.shape[1] == T, "Model is trained to expect " + str(T) + " words, but word-matrix has " + str(W.shape[1]) + " words"
   
    expLmda  = queryState.expLmda;
    row_sums = expLmda.sum(axis=1)
    expLmda /= row_sums[:, np.newaxis] # converts it to a true distribution
    
    likely = np.sum (sparseScalarProductOf(W, safe_log(expLmda.dot(vocab))).data)
    
    # Revert expLmda to its original value as this is a ref to, not a copy of, the original matrix
    expLmda *= row_sums[:, np.newaxis]
    
    return likely

 
def deriveXi (lmda, nu, s):
    '''
    Derives a value for xi. This is not normally needed directly, as we
    normally just work with the negJakkola() function of it
    '''
    return np.sqrt(lmda**2 - 2 * lmda * s[:,np.newaxis] + (s**2)[:,np.newaxis] + nu**2)   

def newVbQueryState(W, K):
    '''
    Creates a new query state for a given document set, which can in this
    case simply be described by the document-term matrix W. The query state
    has all the parameters associated with the inference of topic memberships.
    
    Params:
    W - the DxT document-term matrix for D documents and T possible terms
    K - the number of topics to infer
    
    Returns:
    A named tuple containing
        expLmda - the DxK matrix of the posterior topic means, having been pumped
                  through the exp() function.
        nu      - the DxK matrix of diagonal covariances of the posterior topic
                  distributions for each of the K documents.
        lxi     - The DxK matrix xi is a parameter of the approximation to the
                  soft-mutionax distrib, and lxi is the DxK matrix resulting having
                  applied the negative Jaakkola function to it
        s       - Dx1 matrix of the offsets used for the approximations to the
                  non-conjugate soft-max topic distribution.
        docLen  - the Dx1 vector of the lengths of each of the documents.
    '''
    D = W.shape[0]
    docLen = np.squeeze(np.asarray (W.sum(axis=1))) # Force to a one-dimensional array for np.newaxis trick to work
    
    
    expLmda = np.exp(rd.random((D, K)).astype(DTYPE))
    nu   = np.ones((D, K), DTYPE)
    s    = np.zeros((D,), DTYPE)
    lxi  = negJakkola (np.ones((D,K), DTYPE))
    
    return VbSideTopicQueryState (expLmda, nu, lxi, s, docLen)
    

def newVbModelState(K, Q, F, P, T, featVar = 0.01, topicVar = 0.01, latFeatVar = 1, latTopicVar = 1):
    '''
    Creates a new model state object for a topic model based on side-information. This state
    contains all parameters that o§nce trained can be kept fixed for querying.
    
    The parameters are
    
    K - the number of topics
    Q - the number of latent topics, Q << K
    F - the number of features
    P - the number of latent features in the projected space, P << F
    T - the number of terms in the vocabulary
    topicVar - a scalar providing the isotropic covariance of the topic-space
    featVar - a scalar providing the isotropic covariance of the feature-space
    latFeatVar - a scalar providing the isotropic covariance of the latent feature-space
    latTopicVar - a scalar providing the isotropic covariance of the latent topic-space
    
    
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
    
    Y     = rd.random((Q,P)).astype(DTYPE)
    omY   = latFeatVar * np.identity(P, DTYPE)
    sigY  = latTopicVar * np.identity(Q, DTYPE)
    
    sigT  = topicVar * np.identity(K, DTYPE)
    
    U     = rd.random((K,Q)).astype(DTYPE)
    V     = rd.random((F,P)).astype(DTYPE)
    
    A     = U.dot(Y).dot(V.T)
    varA  = featVar * np.identity(F, DTYPE)
    
    varRatio = (featVar * topicVar) / (latFeatVar * latTopicVar)
    if varRatio > 1:
        raise ValueError ("Model will not converge as (featVar * topicVar) / (latFeatVar * latTopicVar)) = " + str(varRatio) + "  when it needs to be no more than one.")
    
    # Vocab is K word distributions so normalize
    vocab = normalizerows_ip (rd.random((K, T)).astype(DTYPE))
    
    return VbSideTopicModelState(K, Q, F, P, T, A, varA, Y, omY, sigY, sigT, U, V, vocab, topicVar, featVar, latTopicVar, latFeatVar)

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


def sparseScalarProductOfDot(A,B,C, out=None):
    '''
    Calculates A * B.dot(C) where A is a sparse matrix
    
    Retains sparsity in the result, unlike the built-in operator
    
    Note the type of the return-value is the same as the type of
    the sparse matrix A. If this has an integral type, this will
    only provide integer-based multiplication.
    '''
    if out is None:
        out = A.copy()
    out.data[:] = A.data
    out.data *= B.dot(C)[csr_indices(out.indptr, out.indices)]
    
    return out

def sparseScalarProductOf(A,B, out=None):
    '''
    Calculates A * B where A is a sparse matrix
    
    Retains sparsity in the result, unlike the built-in operator
    
    Note the type of the return-value is the same as the type of
    the sparse matrix A. If this has an integral type, this will
    only provide integer-based multiplication.
    '''
    if out is None:
        out = A.copy()
    if not out is A:
        out.data[:] = A.data
    out.data *= B[csr_indices(out.indptr, out.indices)]
    
    return out

def sparseScalarQuotientOfDot(A,B,C, out=None):
    '''
    Calculates A / B.dot(C) where A is a sparse matrix
    
    Retains sparsity in the result, unlike the built-in operator
    
    Note the type of the return-value is the same as the type of
    the sparse matrix A. If this has an integral type, this will
    only provide integer-based multiplication.
    '''
    if out is None:
        out = A.copy()
    if not out is A:
        out.data[:] = A.data
    out.data /= B.dot(C)[csr_indices(out.indptr, out.indices)]
    
    return out


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

