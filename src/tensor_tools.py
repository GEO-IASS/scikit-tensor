import numpy as np

def khatrirao(matrices, reverse=False):
    # Compute the Khatri-Rao product of all matrices in list "matrices".
    # If reverse is true, does the product in reverse order.
    matorder = range(len(matrices)) if not reverse else list(reversed(range(len(matrices))))
    
    # Error checking on matrices; compute number of rows in result.
    # N = number of columns (must be same for each input)
    N = matrices[0].shape[1] 
    # Compute number of rows in resulting matrix
    # After the loop, M = number of rows in result.
    M = 1
    for i in matorder:
        if matrices[i].ndim != 2:
            raise ValueError("Each argument must be a matrix.")
        if N != (matrices[i].shape)[1]:
            raise ValueError("All matrices must have the same number of columns.")
        M *= (matrices[i].shape)[0]
        
    # Computation
    # Preallocate result.
    P = np.zeros((M, N))
    
    # n loops over all column indices
    for n in range(N):
        # ab = nth col of first matrix to consider
        ab = matrices[matorder[0]][:,n]
        # loop through matrices
        for i in matorder[1:]:
            # Compute outer product of nth columns
            ab = np.outer(matrices[i][:,n], ab[:])
        # Fill nth column of P with flattened result
        P[:,n] = ab.flatten()
    return P
