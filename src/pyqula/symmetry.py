import numpy as np
from scipy.optimize import minimize,broyden1
import scipy.linalg as lg


def v2U(v,exp=True):
    """
    Transform a vector into a unitary matrix
    """
    n = int(np.sqrt(v.shape[0]+1)) # dimension of the matrix
    M = np.zeros((n,n),dtype=np.complex) # define the matrix
    ii = 0 # counter
    d = np.zeros(n,dtype=np.complex) # diagonal
    for i in range(n-1):
        d[i] = v[ii]
        ii += 1 # increase counter
    d[n-1] = -np.sum(d) # ensure the trace is zero
    for i in range(n): M[i,i] = d[i] # put the diagonal elements
    for j in range(n):
        for i in range(j+1,n):
            M[i,j] = v[ii] + 1j*v[ii+1] # store
            M[j,i] = v[ii] - 1j*v[ii+1] # store
            ii += 2 # increase
    if exp: return lg.expm(1j*M) # return unitary matrix
    else: return M # return the Lie algebra matrix




def v2O(v,exp=True):
    """
    Transform a vector into an orthogonal matrix
    """
    n = int(np.sqrt(v.shape[0]+1)) # dimension of the matrix
    M = np.zeros((n,n)) # define the matrix
    ii = 0 # counter
    for j in range(n):
        for i in range(j+1,n):
            M[i,j] = v[ii]  # store
            M[j,i] = -v[ii]  # store
            ii += 1 # increase
    if exp: return lg.expm(M).real # return orthogonal matrix
    else: # return the algebra matrix (for continuus symmetries)
      return M
#      return M/np.sqrt(np.trace(M@np.transpose(M))) # return normalized matrix



class SymGen():
    """
    Class for the symmetry generator
    """
    def __init__(self,m,discrete=True):
        self.real = np.abs(np.max(m.imag))<1e-7 # real generator
        self.discrete = discrete # discrete symmetry
        self.np = m.shape[0]**2-1 # number of free variables
        self.n = m.shape[0] # dimension of the operator
    def get_x0(self):
        return np.random.random(self.np) # return an initial value
    def get_symmetry(self,x):
        """Return the symmetry operator"""
        if self.real: 
            M = v2O(x,exp=self.discrete)
            if self.discrete: return M # return the operator
            else: return lg.expm(M) # return the operator
        else:
            M = v2U(x,exp=self.discrete)
            if self.discrete: return M # return the operator
            else: return lg.expm(1j*M) # return the operator
    def get_operator(self,x):
        """Return the operator used to compute the commutator"""
        if self.real: return v2O(x,exp=self.discrete)
        else: return v2U(x,exp=self.discrete)
    def return_unique(self,mm):
        """Return the independent symmetry operators"""
        ms = [] 
        for m in mm: 
            if m is not None: ms.append(m) # store this matrix
        if self.discrete: return retain_different(ms)
        else: 
            vs = [m.reshape(self.n**2) for m in ms] # to vectors
            vs = retain_independent(vs) # independent vectors
            ms = [v.reshape((self.n,self.n)) for v in vs]
            ms = [m/np.max(np.abs(m)) for m in ms]
            return ms






def commuting_matrices(m):
    """
    Compute the different unitary operators
    that commute with a matrix.
    Written for SU(N) and SO(N)
    """
    m = np.array(m) # convert to array
    real = np.max(np.abs(m.imag))<1e-3 # real matrix
    n = m.shape[0] # size of the matrix
    SG = SymGen(m,discrete=False) # Generator for discrete symmetry
    def fun(v): # function to minimize
        U = SG.get_operator(v) # get the matrix
        A = m@U - U@m # commutator
        out = np.trace(A@np.conjugate(A.T)) # return trace
        return out.real
#    if real: v0 = np.random.random((n**2-n)//2) # initial guess
# this should be fixed
    def getU(): # return one symmetry operator
      v0 = SG.get_x0() # initial guess
      res = minimize(fun,v0) # minimize
      v = res.x
      U = SG.get_symmetry(v)
      U = SG.get_operator(v)
#      print(np.round(U,2),"Error",fun(v))
      if fun(v)>1e-5: return None
      else: return U
    Us = [getU() for i in range(100)] # compute many of them
    Us = SG.return_unique(Us)
#    Us = retain_different(Us)
    print("Found",len(Us),"symmetries")
    for u in Us:
        print(np.round(u,2))


def retain_different(us):
    """
    Retain operators that are different
    """
    out = []
    for u in us: # loop over input matrices
        if u is None: continue
        store = True # assume that it is different
        for o in out: # loop over stored ones
            if np.max(np.abs(u-o))<1e-4: store = False # already have it
        if store: out.append(u) # store this one
    return out



def retain_generators(us):
    """
    Retain the matrices that generate the symmetry algebra
    """
    out = [] # output list



def retain_independent(M):
    """Return linearly independent vectors"""
    from numpy.linalg import matrix_rank
    dim = len(M)
    tol = 1e-4
    LI=[]
    for i in range(dim):
        tmp=[]
        for r in LI:
            tmp.append(r)
        tmp.append(M[i])   
        if matrix_rank(tmp,tol=tol)>len(LI):    
            LI.append(M[i])      
    return LI       



