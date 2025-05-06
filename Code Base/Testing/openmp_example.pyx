# openmp_example.pyx

from cython.parallel import prange
cimport openmp

def compute_sum(int n):
    cdef int i
    cdef double result = 0.0

    # Parallel loop with OpenMP
    # prange will release the GIL itself when nogil=True
    for i in prange(n, nogil=True, num_threads=4):  # Use 4 threads
        result += i  # Summing the numbers in parallel

    return result
