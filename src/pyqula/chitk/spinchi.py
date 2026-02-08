import numpy as np
from .rpa import chi_AB_RPA

# spin-response functions


def spinchi_ladder(H,v=[0.,0.,1.],RPA=True,**kwargs):
    """Return the spin response function"""
    if H.has_eh: 
        print("Not implemented with Nambu basis")
        raise
    sx = H.get_operator("sx") # spin operator, eigen +-1
    sy = H.get_operator("sy") # spin operator, eigen +-1
    sz = H.get_operator("sz") # spin operator, eigen +-1
    v = np.array(v) # convert to array
    # this is not finished yet
    sp = (sx + 1j*sy)/2. # ladder operator
    sm = (sx - 1j*sy)/2. # ladder operator
    U = H.V # get the interaction
    if U is not None: # finite interaction
        U = U[(0,0,0)] # onsite interaction matrix
        U = sp@U # project on the operator
        U = V2U_matrix(U) # transform the U matrix (2N) into the (N)
        U = -2*U # beware of this minus sign for spin response (!!!)
    if not RPA: U = None # no RPA
    return chi_AB_RPA(H,A=sp,B=sm,V=U,**kwargs) # non-interacting response



def V2U_matrix(V):
    """Transform the V interaction into the U matrix needed for RPA"""
    # V is a 2N matrix with individual U values in the diagonal
    N = V.shape[0]//2 # dimension
    U = np.zeros((N,N),dtype=np.complex128) # initialize
    for i in range(N):
        U[i,i] = V[2*i,2*i] # add contribution
        U[i,i] += V[2*i+1,2*i+1] # add contribution
    return U # return the matrix


