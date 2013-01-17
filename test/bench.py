from time import clock
import tensor
import ktensor
import tensor_tools
import numpy as np
from numpy import ndarray, unravel_index, prod


def mmat(x, format='%.12e'):
    """Display the ndarray 'x' in a format suitable for pasting to MATLAB"""

    def print_row(row, format):
        for i in row:
            print format % i,

    if x.ndim == 1:
        # 1d input
        print "[",
        print_row(x, format)
        print "]"
        print ""

    if x.ndim == 2:
        print "[",
        print_row(x[0], format)
        if x.shape[0] > 1:
            print ';',
        for row in x[1:-1]:
            print " ",
            print_row(row, format)
            print ";",
        if x.shape[0] > 1:
            print " ",
            print_row(x[-1], format)
        print "]",

    if x.ndim > 2:
        d_to_loop = x.shape[2:]
        sls = [slice(None,None)]*2
        print "reshape([ ",
        # loop over flat index
        for i in range(prod(d_to_loop)):
            # reverse order for matlab
            # tricky double reversal to get first index to vary fastest
            ind_tuple = unravel_index(i,d_to_loop[::-1])[::-1]
            ind = sls + list(ind_tuple)
            mmat(x[ind],format)          

        print '],[',
        for i in x.shape:
            print '%d' % i,
        print '])'

# bench TTT times
TTT_times = [0.0]*15
for i in range(1, 16):
    A = tensor.tenrands((i,i,i))
    B = tensor.tenrands((i,i,i))
    start = clock()
    C = A.ttt(B)
    stop = clock()
    TTT_times[i-1] = stop-start
print ",".join([str(elt) for elt in TTT_times])

# bench CP_ALS times
CP_ALS_times = np.zeros((6, 5))
for i in range(3,9):
    A = tensor.Tensor(np.arange(i ** 3), (i,i,i))
    for j in range(1,6):
        tic = clock()
        p = A.cp_als(j, tol=1e-9, maxiters=1000, printitn=1000)
        toc = clock()
        CP_ALS_times[i-3,j-1] = toc - tic
print mmat(CP_ALS_times)

