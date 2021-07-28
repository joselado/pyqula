import numpy as np




class Tensor3():
    """Class for a three dimensional tensor"""
    def __init__(self,row,col,pla,data,shape=None):
        if shape is None: raise
        self.row = row
        self.col = col
        self.pla = pla
        self.data = data
        self.shape = shape
    def __mul__(self,a):
        """Multiply by a vector, assuming that it will use
        the third component, and return an array"""
        if a.shape[0]!= self.shape[2]: raise # inconsistent dimensions
        out = np.zeros((self.shape[0],self.shape[1]),dtype=np.complex)
        for ii in range(len(self.row)):
            i = self.row[ii]
            j = self.col[ii]
            k = self.pla[ii]
            d = self.data[ii]
            out[i,j] += a[k]*d # add contribution
        return out # return matrix
    def copy(self):
        from copy import deepcopy
        return deepcopy(self)
    def __add__(self,a):
        """Sum two tensors"""
        if type(a)!=Tensor3: raise # not implemented
        if a.shape[0]!=self.shape[0]: raise
        if a.shape[1]!=self.shape[1]: raise
        if a.shape[2]!=self.shape[2]: raise
        out = self.copy() # copy object
        out.row = np.concatenate([self.row,a.row])
        out.col = np.concatenate([self.col,a.col])
        out.pla = np.concatenate([self.pla,a.pla])
        out.data = np.concatenate([self.data,a.data])
        return out # return object



class TensorN():
    """Define an N dimensional sparse tensor"""
    def __init__(self,index,data,shape=None):
        self.index = index
        self.data = data
        self.shape = shape




