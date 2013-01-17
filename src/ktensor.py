import numpy as np
import tensor
from math import sqrt
import tensor_tools as tt

class KTensor(object):
    # A representation of a tensor as a Kruskal operator (decomposed form).
    def __init__(self, u_list, lmbda=None):
        if len(u_list) == 0:
            raise ValueError("Must include at least one matrix.")
        if lmbda is None:
            # Assume lambda is a k-length column vector (each U_m is a matrix with k columns) 
            # with each entry equal to one.
            lmbda = np.ones(u_list[0].shape[1], 1)
        
        # Error checking
        # Make sure each U_m is a matrix.
        if not all([u.ndim == 2 for u in u_list]):
            raise ValueError("Not every Um is a matrix.")
        # Check for size errors - make sure each matrix has k columns.
        k = len(lmbda)
        if not all([u.shape[1] == k for u in u_list]):
            raise ValueError("Not every Um has %d columns." % k)
           
        self.lmbda = lmbda
        self.u_list = u_list

    def shape(self):
        # The shape of the k-tensor is a list of the number of rows in each u matrix
        return tuple([u.shape[0] for u in self.u_list])

    def ndim(self):
        # The number of dimensions in the k-tensor.
        return len(self.shape())
        
    def norm(self):
        # The frobenius norm of the tensor.
        coeff = np.outer(self.lmbda, self.lmbda.T)
        for i in range(self.ndim()):
            coeff = coeff * np.dot(self.u_list[i].T, self.u_list[i])
        return sqrt(abs(np.sum(coeff.flatten())))

    def full(self):
        # Convert to a full dense tensor.
        A = tt.khatrirao(self.u_list, reverse=True)
        data = np.dot(self.lmbda.T, A.T)
        return tensor.Tensor(data, self.shape())

    def __str__(self):
        return "KTensor of shape %s\n Lambda:%s" % (self.shape(), str(self.lmbda)) + "\n Factor Matrices:\n" + "\n\n".join([str(u) for u in self.u_list])
