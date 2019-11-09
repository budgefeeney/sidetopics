

cdef double l1_dist_f64 (double[:] left, double[:] right) nogil

cdef inline void initAtRandom_f64(double[:,:] topicDists, int d, int K) nogil