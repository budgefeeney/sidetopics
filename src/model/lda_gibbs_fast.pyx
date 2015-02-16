cimport cython
import numpy as np
cimport numpy as np

from libc.stdint cimport uint16_t, uint8_t
from libc.stdlib cimport srand, rand, RAND_MAX

# Import the random number generation functions from
# the GNU Scientific Library
# cdef extern from "gsl/gsl_rng.h":
#     ctypedef struct gsl_rng_type
#     ctypedef struct gsl_rng
#     
#     cdef gsl_rng_type *gsl_rng_mt19937
#     
#     gsl_rng *gsl_rng_alloc ( gsl_rng_type * T) nogil
#     void gsl_rng_free (gsl_rng * r) nogil
#     
#     void gsl_rng_set ( gsl_rng * r, unsigned long int seed) nogil
#     
#     double gsl_rng_uniform ( gsl_rng * r) nogil
    
# We just set up a single, non-threadsafe, global RNG
# This must be manually freed to avoid craashing the Python interpreter
# cdef gsl_rng *global_rng = gsl_rng_alloc(gsl_rng_mt19937)
 
def initGlobalRng (int randSeed):
    srand(randSeed)
#     gsl_rng_set(global_rng, randSeed)
 
def freeGlobalRng (int randSeed):
    pass
#     gsl_rng_free(global_rng)


    
@cython.cdivision(True)
cdef int rand_lim(int min_val, int max_val) nogil:
    '''
    return a random number between 0 and limit inclusive.
    '''
    return <int> ((rand() / (RAND_MAX + 1.0)) \
                  * (max_val - min_val+1) + min_val)
    


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double[:] inferPolyaHyper (int[:,:] counts, int[:] countsSum):
    '''
    Given draws from a Polya (i.e. Dirichlet multinomial)
    distribution, infer its parameters.
    
    counts    - the N x K matrix of N samples of K-dimensional datapoints
    countsSum - counts.sum(axis=1)
    '''
    cdef:
        double    minChange, change
        double[:] soln,      newSoln
        int       itr
    
    minChange = 0.01 / counts.shape[1]
    
    soln  = np.mean(counts, axis = 0)
    soln  = np.divide(soln, np.sum(soln))
    
    change = minChange + 1
    itr    = 0
    
    while change > minChange and itr < 1000:
        itr     += 1 
        newSoln  = inferPolyaHyperSingle(counts, countsSum, soln)
        change   = np.sum (np.abs (np.subtract(soln, newSoln)))
        soln[:]  = newSoln
    
    return np.array(soln)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double[:] inferPolyaHyperSingle(int[:,:] counts, int[:] countsSum, double[:] old):
    '''
    Given draws from a Polya (i.e. Dirichlet multinomial)
    distribution, infer its parameters. 
    
    Note this requires several subsequent calls to iteratively
    improve the estmate passed in via the "old" parameter
    
    counts    - the N x K matrix of N samples of K-dimensional datapoints
    countsSum - counts.sum(axis=1)
    old       - the old value of the parameter, set to none
                to make an initial estimate.
    '''
    
    cdef:
        int d, D = counts.shape[0]
        int k, K = counts.shape[1]
        double[:] new
        double    denom
        double    oldSum
    
    new    = np.zeros((K,), dtype=np.float64)
    oldSum = np.sum(old)
    
    with nogil:
        for d in range(D):
            for k in range(K):
                new[k] += counts[d,k] / (counts[d,k] - 1 + old[k])
            denom += countsSum[d] / (countsSum[d] - 1 + oldSum)
            
        for k in range(K):
            new[k] = old[k] * new[k] / denom \
                   + 0.00001 # make sure it's never zero in case counts[d,k] = 1
            
    return new
    

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sumSuffStats(\
        uint16_t[:] w_list, \
        uint8_t[:]  z_list, \
        int[:]   docLens,   \
        int[:,:] ndk,       \
        int[:,:] nkv,       \
        int[:]   nk):
    '''
    Sum up the topic assignments into the given
    count matrices ndk, nkv, nk, which should be
    set to zero
    '''
    cdef:
        int n = -1, k, v
        int D = ndk.shape[0], K = ndk.shape[1], T = nkv.shape[1]
    
    for d in range(D):
        for _ in range(docLens[d]):
            n += 1
            
            v = w_list[n]
            k = z_list[n]
            
            ndk[d,k] += 1
            nkv[k,v] += 1
            nk[k]    += 1


  

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sample ( \
        int count,   \
        int thin,    \
        uint16_t[:] w_list,  \
        uint8_t[:]  z_list,  \
        int[:]      docLens, \
        int[:,:]    ndk, \
        int[:,:]    nkv, \
        int[:]      nk,  \
        np.ndarray[np.float64_t, ndim=2] topicSum,   \
        np.ndarray[np.float64_t, ndim=2] vocabSum,   \
        double[:]   a, \
        double[:]   b, \
        bint        isQuery):
    '''
    Uses a collapsed Gibbs sampler to infer the parameters for an
    LDA model
    
    Parameters are updated in-place
    
    count      - total number of samples to draw
    thin       - only evert thin-th sample will be stored in topicSum, vocabSum
    w_list     - the list of word identifiers
    z_list     - the corresponding list of topic identifiers
    docLens    - the lengths of every document, helps partition the lists above
    ndk        - the DxK matrix of how many times each topic has been assigned 
                 to each document
    nkv        - the KxT matrix of how many times each word has been generated
                 by each topic
    nk         - the K-dim vector of how many words have been generated by each
                 topic
    topicSum   - the DxK sum of topic samples, the actual result of the sampler
                 used to generate the topic distributions
    vocabSum   - likewise but for vocabulary
    a          - the prior over topics, a vector
    b          - the prior over vocabulary, a vector
    isQuery    - if true the vocabulary distribution is kept fixed
    '''
    cdef:
        int s, d, n, nsimple, ncount, j
        int i, start = -docLens[0]
        int D = ndk.shape[0], K = ndk.shape[1], T = nkv.shape[1]
        int k, v
        int trueSampleCount = 0
        int queryDelta
        double[:] dist
        double    distNorm, draw
        double    aSum, bSum
    
    aSum = np.sum(a)
    bSum = np.sum(b)
    dist = np.empty((K,), dtype=np.float64)
    queryDelta = 0 if isQuery else 1
    
    with nogil:
        for s in range(count):
            start = 0
            for d in range(D):
                # we randomise the point at which we start looping
                # through words in documents
                nsimple = rand_lim(0, docLens[d] - 1) #<int> (gsl_rng_uniform (global_rng) * docLens[d])
                for ncount in range(docLens[d]):
                    nsimple = (nsimple + 1) % docLens[d]
                    n = start + nsimple
                    
                    # Current word and its topic
                    k = z_list[n]
                    v = w_list[n]

                    # Remove current word's topic from suff stats
                    ndk[d,k] -= 1
                    nkv[k,v] -= queryDelta
                    nk[k]    -= 1

                    # Generate a distribution over topics
                    distNorm = 0.0
                    for j in range(K):
                        dist[j] = (ndk[d,j] + a[j]) \
                                * (nkv[j,v] + b[v]) \
                                / (nk[j] + bSum)
                        distNorm += dist[j]

                    # Choose a new topic for the current word
                    k = -1
                    draw = (<double> rand()) / RAND_MAX * distNorm #gsl_rng_uniform (global_rng) * distNorm
                    while draw > 0:
                        k = (k + 1) % K # shouldn't need mod, but to be safe...
                        draw -= dist[k]
                        
                    z_list[n] = k
                    
                    # Add current word's new topic to suff stats
                    ndk[d,k] += 1
                    nkv[k,v] += queryDelta
                    nk[k]    += 1

                start += docLens[d]
            # Check if this is one of the samples to take
            if (s + 1) % thin == 0:
                with gil:
                    print ("\nTaking sample " + str(trueSampleCount))
                    
                # Avoid fully vectorising so we don't materialise huge, temporary, DxK or KxV matrices
                # TODO Consider doing this a row at a time, using numpy, for speedup
                for d in range(D):
                    for k in range(K):
                        topicSum[d,k] += (ndk[d,k] + a[k]) / (docLens[d] + aSum)
                for k in range(K):
                    for v in range(T):
                        vocabSum[k,v] += nkv[k,v] + b[v] / (nk[k] + bSum)
                
                trueSampleCount += 1
                    
#                 if trueSampleCount % 10 == 0:
#                     with gil:
#                         print ("Updating hyperparameters...")
#                         a[:] = inferPolyaHyper(ndk, docLens)
#                         b[:] = inferPolyaHyper(nkv, np.sum(nkv, axis=1, dtype=np.int32))
#                        
#                         aSum = np.sum(a)
#                         bSum = np.sum(b)
#                         print ("Done")

            
    return trueSampleCount
 

def flatten (sparseMat):
    '''
    Takes a DxT sparse CSR matrix X indicating how often the
    identifier t in 1..T occurred, and flattens it into a sequence
    of identifiers which has X.sum() elements total.
    
    The elements are shuffled, so we don't observe the same
    element all the time.
    
    Returns
     - A single array with X.sum() elements each in the range
       [0,T)
     - A list of row lengths, X.sum(axis=1)
    '''
    (D,T)   = sparseMat.shape
    if T > 65536:
        raise ValueError("Too many words for unit16 storage")
    docLens = np.squeeze(np.asarray(sparseMat.sum(axis=1)))
    out     = np.ndarray(shape=(docLens.sum(),), dtype=np.uint16)
    
    flatten_native (sparseMat.indices, sparseMat.data, sparseMat.indptr, out)
    
    return out, docLens
    

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void flatten_native(int[:] indices, int[:] data, int[:] indptr, uint16_t[:] out) nogil:
    '''
    A low-level implementation of flatten(sparseMat), see
    that method for more information.
    '''
    cdef:
        int i, j, o = 0
    
    # Unwrap the array
    for i in range(indices.shape[0]):
        for j in range(data[i]):
            out[o] = indices[i]
            o += 1
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void shuffle (uint16_t[:] a, int start, int end) nogil:
    '''
    Shuffles the elements of the array in the given range, 
    leaving elements outside that range as they were before.
    '''
    cdef:
        int left, right, tmp
        int i, span, shuffle_count
    
    srand(0xC0FFEE + a[start])
    shuffle_count = end - start / 2
    
    for i in range(shuffle_count):
        left  = rand_lim(start, end - 1)
        right = rand_lim(start, end - 1)
        
        tmp      = a[left]
        a[left]  = a[right]
        a[right] = tmp
    


