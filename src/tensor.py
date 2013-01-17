import numpy as np
import tensor_tools as tt
import itertools
import operator
import tenmat
import ktensor

# A class that represents a dense tensor of arbitrary rank. 
# The Tensor class wraps a numpy.ndarray object and provides implementations of the following:
#  - size: returns the number of elemnts in the tensor
#  - ndim: returns the number of dimensions (rank) of the tensor
#  - permute(order): returns a new Tensor with its data permuted according to order
#  - inv_permute(order): returns a new Tensor with its data permuted 
#    such that T.permute(order).inv_permute(order).data = T.data
#  - tensordot(other): returns the k-fold innerproduct of self with other tensor (returns a scalar)
#  - mttrkp(U, n): returns the matricized tensor times Khatri-Rao product 
#    of self with all matrices (except the nth) in the list
#  - ttv(v, n): returns the result of tensor-times-vector in the nth mode.
#  - ttv_list(uList, nList): returns the result of sequentially doing each 
#    ttv product for pairs (u, n).
#  - ttm(A, n): returns the result of tensor-tiems-matrix in the nth mode.
#  - ttm_list(AList, nList): returns the result of sequentially doign each 
#    ttm product for pairs (A, n).
#  - ttt(other, xDims, yDims): returns the outer product of self with other
#    in the dimensions specified.
#  - cp_als(rank, tolerance, max_iters, print_iters): returns a KTensor object corresponding to
#    the CANDECOMP/PARAFAC decomposition of self.


# Specialzied (convenience) constructors.
def tenones(shape):
    # Tensor with all values = 1
    return Tensor(np.ones(shape), shape)
    
def tenzeros(shape):
    # Tensor with all values = 0
    return Tensor(np.zeros(shape), shape)
    
def tenrands(shape):
    # Tensor with values chosen uniformly at random.
    return Tensor(np.random.random(shape), shape)
        
class Tensor(object):
    def __init__(self, data, shape=None):
        # Constructor for dense tensor. 
        # Data can be of type numpy.array or of type list.
        # Shape can be of type numpy.array, list, or tuple.
        if type(data) == list:
            data = np.array(data)
        if shape != None:
            if len(shape) == 0:
                raise ValueError("Shape must be non-empty.")
            if type(shape) == np.ndarray and shape.ndim != 2 and shape[0].size != 1:
                raise ValueError("Shape must be a non-empty row vector.")
            # If passed an explicit shape, use it; otherwise use the shape of the data.
            shape = tuple(shape)
        else:
            shape = tuple(data.shape)
            
        if len(shape) == 0: 
            if data.size != 0:
                raise ValueError("Empty tensor by shape cannot have elements.")
        elif np.prod(shape) != data.size:
            raise ValueError("Tensor shape does not match tensor size.")
        data = np.reshape(data, shape)
        self.shape = shape
        self.data = data
        
    def size(self):
        # Return the number of elements in tensor.
        return self.data.size
        
    def ndim(self):
        # Return the number of dimensions of tensor (rank).
        return self.data.ndim
         
    def __str__(self):
        # String representation of tensor.
        return str(self.data)
        
    def copy(self):
        # Full deep copy of the tensor.
        return Tensor(self.data.copy(), self.shape)

    def permute(self, order):
        # Return a new tensor object permuted according to order.
        # The higher-dimensional analog of transpose.
        new = np.transpose(self.data, order)
        return Tensor(new, new.shape)
        
    def inv_permute(self, order):
        # Return a tensor permuted by the INVERSE of order.
        new = np.transpose(self.data, np.argsort(order))
        return Tensor(new, new.shape)
        
    def tensordot(self, other):
        # Returns k-fold innerproduct of self with other (np.tensordot over all axes).
        if type(other) == Tensor:
            if self.data.ndim != other.data.ndim:
                print "self.data.ndim = %s, other.data.ndim = %s" % (self.data.ndim, other.data.ndim)
                raise ValueError("Cannot compute this dot product, sizes don't match.")
            return np.tensordot(self.data, other.data, axes = (range(self.data.ndim), range(other.data.ndim)))
        elif type(other) == ktensor.KTensor:
            return self.tensordot(other.full())
            
    def mttkrp(self, U, n):
        # Compute the matrix product of the n-mode matricization of self with the KR
        # product of all entries in U, a list of matrices, except the nth one.
        X = self.data
        N = X.ndim
        order = range(N)
        order_sans_n = order[:]; order_sans_n.remove(n)
        Xn = X.transpose([n] + order_sans_n)
        Xn = Xn.reshape(X.shape[n], X.size / X.shape[n])
        Z = tt.khatrirao([U[i] for i in order_sans_n], reverse=True)
        return np.dot(Xn, Z)
        
    ###############################
    # Tensor times various things #
    ###############################
    def ttv(self, u, n):
        # Compute the n-mode product (Kolda-style) of tensor self with vector u
        # X.ttv(u, n) computes the n-mode product of tensor X with a vector u; 
        # i.e., X x_n u. The integer n specifies the dimension (or mode) of X along which u
        # should be contracted.
        
        # Given X = I_0 x I_1 x ... x I_n x ... I_N
        # Given u an I_n vector
        # X x_n u = I_0 x I_1 x ... x I_n-1 x I_n+1 x ... I_N
        
        # Computed with a clever abuse of notation (np.einsum).
        if self.shape[n] != u.shape[0]:
            raise ValueError("Cannot compute this tensor-vector product because dimensions don't match.")
        else:
            first = "".join([chr(ord('i') + k) for k in range(self.data.ndim)])
            second = first[n]
            third = first[:]
            third = third[:n] + third[n+1:]
            einstring = first+","+second+"->"+third
            return Tensor(np.einsum(einstring, self, u))

    def ttv_list(self, uList, nList):
        # Do ttv on each matrix in uList sequentially.
        result = self.copy()
        for (u, n) in zip(uList, nList):
            result = result.ttv(u, n)
        return result
     
    def ttm(self, A, n, option=None):
        # Compute the n-mode product (Kolda-style) of tensor self with matrix A
        # X.ttm(A, n) computes the n-mode product of tensor X with matrix A;
        # i.e., X x_n A. The integer n specifies the dimension (or mode) of X along which A
        # should be contracted.
        
        # Computed with a clever abuse of notation (np.einsum).
        if self.shape[n] != A.shape[1]:
            raise ValueError("Cannot compute this tensor-matrix produdct because dimensions don't match.")
        else:
            if option == 't':
                A = A.transpose()
            # sample string - mode-1 multiplication
            # np.einsum("ijk,aj->iak", self, A)
            first = "".join([chr(ord('i') + k) for k in range(self.data.ndim)])
            second = "a"+first[n]
            third = first[:]
            third = third[:n] + "a" + third[n+1:]
            einstring = first+","+second+"->"+third
            return Tensor(np.einsum(einstring, self, A))

    def ttm_list(self, AList, nList, option=None):
        # Do ttm on each matrix in AList sequentially.
        result = self.copy()
        for (A, n) in zip(AList, nList):
            result = result.ttm(A, n, option)
        return result
     
    def ttt(self, Y, xdims=[], ydims=[]):
        # Compute the outer product of two tensors X (self) and Y.
        # If xdims is present, compute the inner product of X and Y in the dimensions specified
        # by xdims and ydims. The sizes of the dimensions specified by xdims and ydims must match,
        # that is, X.shape[xdims[i]] = y.shape[ydims[i]] for all i
        # Convert to matricized storage (avoid transpose by reshaping A and computing C = A * B
        if xdims:
            if not all([self.shape[xdims[i]] == Y.shape[ydims[i]] for i in range(len(xdims))]):
                raise ValueError("Tensor lengths do not match along specified dimensions.")
        s = tenmat.Tenmat(self, cdim=xdims, option='t')
        y = tenmat.Tenmat(Y, rdim=ydims)
        
        cdata = np.dot(s.data, y.data)
        crind = range(len(s.rindices))
        ccind = range(len(s.rindices), len(s.rindices)+len(y.cindices))
        ctsiz = [s.tsize[i] for i in s.rindices] + [y.tsize[i] for i in y.cindices]
        
        c = tenmat.Tenmat(cdata, crind, ccind, ctsiz)
        return c.to_dense_tensor()
        
    
    def cp_als(self, R, tol=1.0e-4, maxiters=50, printitn=10):
        # Compute the rank-R CANDECOMP/PARAFAC decomposition 
        # via alternating least squares. 
        # Stop when the change in the fit is less than tol.
        # Print every printitn iterations.
        N = self.data.ndim
        normSelf = np.linalg.norm(self.data)
        dimorder = range(N)
        
        Uinit = []
        for n in dimorder:
            Uinit.append(np.random.rand(self.shape[n], R))
        
        # Set up for iterations - initialize U and the fit.
        U = Uinit
        fit = 0
        if printitn > 0:
            print "CP_ALS:"
        
        for iter in range(maxiters):
            fitold = fit
            for n in dimorder:
                # Calculate Unew = X_n * khatrirao(all U except n, reverse=True)
                Unew = self.mttkrp(U, n)
                
                # Compute the matrix of coefficients for linear system
                Y = np.ones((R, R))
                c = range(N); c.remove(n)
                for i in c:
                    Y = Y * np.dot(U[i].T, U[i])                   
                # Solve.
                Unew = np.dot(Unew, np.linalg.pinv(Y))
                
                # Normalize each vector to prevent singularities in coefficient matrix
                if iter == 0:
                    # lmbda is 2-norm of each column
                    lmbda = np.sqrt(np.sum(Unew * Unew, 0))
                else:
                    # lmbda is max-norm of each column
                    lmbda = np.max(Unew, 0)
                    
                Unew = np.dot(Unew, np.diag(1 / lmbda))
                U[n] = Unew
            
            # Construct our approximation
            P = ktensor.KTensor(U, lmbda)
            
            # Compute norm of residual
            normresidual = np.sqrt(normSelf**2.0 + P.norm() ** 2.0 - 2.0*(self.tensordot(P)))
            
            # Figure out how good the fit is
            fit = 1 - (normresidual / normSelf) # Fraction explained by the model
            fitchange = abs(fitold - fit)
            
            if iter % printitn == 0:
                print "Iter %2d: fit = %e fitdelta = %7.1e\n" % (iter, fit, fitchange)
            
            if iter > 1 and fitchange < tol:
                break
        return P

            
        
    
        
        
        
        
    
        
    
