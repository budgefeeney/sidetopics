'''
The inputs and outputs of a SideTopicModel

Created on 29 Jun 2013

@author: bryanfeeney
'''

from collections import namedtuple
import numpy as np
import numpy.linalg as la
import numpy.random as rd
import scipy as sp
import scipy.sparse as ssp

# TODO Consider using numba for autojit (And jit with local types)
# TODO Investigate numba structs as an alternative to namedtuples
# TODO Make random() stuff predictable, either by incorporating a RandomState instance into model parameters
#      or calling a global rd.seed(0xCAFEBABE) call.

VbSideTopicQueryState = namedtuple ( \
    'VbSideTopicState', \
    'X W lmda nu lxi s'\
)


def negJakkola(vec):
    '''
    The negated version of the Jakkola expression which was used in Bourchard's NIPS
    2007 softmax bound
    '''
    
    # COPY AND PASTE BETWEEN THIS AND negJakkolaOfDerivedXi()
    return 1./(2*vec) * (1./(1 - np.exp(-vec)) - 0.5)

def negJakkolaOfDerivedXi(d, lmda, nu, s):
    '''
    The negated version of the Jakkola expression which was used in Bourchard's NIPS
    2007 softmax bound calculating using the estimate of xi using lambda, nu, and
    s
    
    d    - the document index (for lambda and nu)
    lmda - the DxK matrix of means of the topic distribution for each document
    nu   - the DxK the vector of variances of the topic distribution
    s    - The Dx1 vector of offsets.
    '''
    
    # COPY AND PASTE BETWEEN THIS AND negJakkola()
    vec = (np.sqrt (lmda[d,:] ** 2 -2 *lmda[d,:] * s[d] + s**2 + nu**2))
    return 1./(2*vec) * (1./(1 - np.exp(-vec)) - 0.5)

def train(modelState, X, W, iterations=1000, epsilon=0.001):
    '''
    Creates a new query state object for a topic model based on side-information. 
    This contains all those estimated parameters that are specific to the actual
    date being queried - this must be used in conjunction with a model state.
    
    The parameters are
    
    modelState - the model state with all the model parameters
    X - the D x F matrix of side information vectors
    W - the D x V matrix of word **count** vectors.
    
    This returns a tuple of new model-state and query-state. The latter object will
    contain X and W and also
    
    s      - A D-dimensional vector describing offset in our bound on the true value of ln sum_k e^theta_dk 
    lxi    - A DxK matrix used in the above bound, containing the negative Jakkola function applied to the 
             quadratic term xi
    lambda - the topics we've inferred for the current batch of documents
    nu     - the variance of topics we've inferred (independent)
    '''
    # Unpack the model state tuple for ease of use and maybe speed improvements
    (K, F, V, P, A, varA, V, varV, U, sigma, tau, vocab) = (modelState.K, modelState.F, modelState.V, modelState.P, modelState.A, modelState.varA, modelState.V, modelState.varV, modelState.U, modelState.sigma, modelState.tau, modelState.vocab)
       
    # We'll need the total word count per document
    docLen = W.sum(axis=1)
    
    # Assign initial values to the query parameters
    D    = np.size(W, 0)
    lmda = rd.random((D, K))
    nu   = np.ones((D,K), np.float32)
    s    = np.zeros((D, 1))
    lxi  = negJakkola (np.ones((D, K), np.float32))
    
    halfSig2 = 1./(sigma*sigma)
    
    # Inference Step 1: Update local parameters given model parameters
    # lmda_dk. rho is DxK
    pred = halfSig2 * np.dot(X, A)
    Z    = normalizerows (np.exp(lmda))
    like = (np.dot(W, np.transpose(vocab)) * Z) / docLen[:, np.newaxis]
    rho  = 2 * s * lxi -0.5 - like
    
    lmda = la.inv()
    
    
    like = np.dot (W, np.transpose())
    for d in xrange(D):
        pred = halfSig2 * dot(A, x)
        
    
    
    # Inference Step 2: Update model parameters given local parameters
    
    return (modelState, VbSideTopicQueryState(X, W, lmda, nu, lxi, s))



VbSideTopicModelState = namedtuple ( \
    'VbSideTopicState', \
    'K F V P A varA V varV U sigma tau vocab'\
)

def newVbModelState(K, F, V, P):
    '''
    Creates a new model state object for a topic model based on side-information. This state
    contains all parameters that once trained can be kept fixed for querying.
    
    The parameters are
    
    K - the number of topics
    F - the number of features
    P - the number of features in the projected space, P << F
    V - the number of words in the vocabulary
    
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
    tau   = 0.001
    sigma = 0.001
    
    # Vocab is K word distributions so normalize
    vocab = normalizerows (rd.random(K, V))
    
    
    return VbSideTopicModelState(K, F, V, P, A, varA, V, varV, U, sigma, tau, vocab)

def normalizerows (matrix):
    row_sums = matrix.sum(axis=1)
    matrix   /= row_sums[:, np.newaxis]
    return matrix

