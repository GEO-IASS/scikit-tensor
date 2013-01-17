import numpy as np
import tensor

class Tenmat:
    # A class that stores a tensor as a matrix.
    def __init__(self, T, rdim=None, cdim=None, tsiz=None, option=None):
        if rdim != None and type(rdim) == list:
            rdim = np.array(rdim)
        if cdim != None and type(cdim) == list:
            cdim = np.array(cdim)
        if tsiz != None and type(tsiz) == list:
            tsiz = np.array(tsiz)  
        
        # First case: we fully specify all params.
        if rdim != None and cdim != None and tsiz != None:
            if type(T) == np.ndarray:
                self.data = T.copy()
            if type(T) == tensor.Tensor:
                self.data = T.data.copy()
            self.rindices = rdim
            self.cindices = cdim
            self.tsize = tuple(tsiz)
            n = len(self.tsize)
            temp = np.concatenate((self.rindices, self.cindices))
            temp.sort()
            if not (np.arange(n) == temp).all():
                raise ValueError("Dimensions specified wrong.")
            elif np.prod(getelts(self.tsize, self.rindices)) != len(self.data):
                raise ValueError("size(T, 0) does not match size specified.")
            elif np.prod(getelts(self.tsize, self.cindices)) != len(self.data):
                raise ValueError("size(T, 1) does not match size specified.")
            return
        if rdim == None and cdim == None:
            raise ValueError("Must specify at least one of rdim or cdim.") 
            
        # Second case: convert a tensor to a tenmat.
        T = T.copy()
        self.tsize = T.shape
        n = T.ndim()    
        if rdim != None:
            if cdim != None:
                rdims = rdim
                cdims = cdim
            elif option != None:
                # User specified an option
                if option == 'fc':
                    # Forward cyclic case - see Kolda
                    rdims = rdim
                    if rdims.size != 1:
                        raise ValueError("Only one row dimension for fc option.")
                    cdims = np.array(range(rdim[0]+1, n) + range(0, rdim[0]))
                elif option == 'bc':
                    # Backward cyclic case - see Kolda
                    rdims = rdim
                    if rdims.size != 1:
                        raise ValueError("Only one row dimension for bc option.")
                    cdims = np.array(range(0, rdim[0])[::-1] + range(rdim[0]+1, n)[::-1])
                else:
                    raise ValueError("Unspecified option for tenmat constructor.")
            else:
                # User did not specify an option
                rdims = rdim
                cdims = np.array([i for i in range(n) if i not in rdims])
        else:
            if option == 't':
                cdims = cdim
                rdims = np.array([i for i in range(n) if i not in cdims])
            else:
                raise ValueError("Unknown option for tenmat constructor.")               
        # Sanity check.
        temp = np.concatenate((rdims, cdims))
        temp.sort()
        if not (np.arange(n) == temp).all():
            raise ValueError("Error, dimensions specified incorrectly.")
        T = T.permute(np.concatenate((rdims, cdims)))
        row = np.prod(getelts(self.tsize, rdims))
        col = np.prod(getelts(self.tsize, cdims)) 
        self.data = T.data.reshape([row, col])
        self.rindices = rdims
        self.cindices = cdims
        
    def copy(self):
        # Return a deep copy of this Tenmat object.
        return Tenmat(self.data, self.rindices, self.cindices, self.tsize)
        
    def to_dense_tensor(self):
        # Return a dense Tensor that corresponds to this Tenmat object.
        order = np.concatenate((self.rindices, self.cindices)).tolist()
        data = self.data.reshape([self.tsize[i] for i in order])
        data = tensor.Tensor(data).inv_permute(order).data
        return tensor.Tensor(data, self.tsize)
        
    
def getelts(iterable, indices):
    # Helper method for pulling out all elements in iterable at each index.
    ret = [];
    for i in indices:
        ret.extend([iterable[i]]);
    return np.array(ret);
                