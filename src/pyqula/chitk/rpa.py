import numpy as np
from .. import algebra

# compute general RPA response function
def chi_AB_RPA(h,V=None,**kwargs):
    """Compute the RPA chi for a hamiltonian"""
    from ..chi import chiAB # get response function
    es,chis = chiAB(h,mode="matrix",**kwargs) # non-interacting response
    iden = np.identity(chis[0].shape[0],dtype=np.complex128) # identity
    if V is not None: # finite interaction, RPA summation
        chis_rpa = [chi@algebra.inv(iden - V@chi) for chi in chis]
    else: chis_rpa = chis
    return es,np.array(chis_rpa)


mode_rpa = "vectorized"

def chi_ops_RPA(h,ops=None,V=None,**kwargs):
    """Compute the RPA chi for a hamiltonian,
    return a tensor given a list of operators. This is 
    for example useful to compute the full spin response
    function"""
    from ..chi import chiAB # get response function
    nop = len(ops) # number of operators
    # storage for the full response
    if mode_rpa=="sequential": # one by one
        chis = [[None for i in range(nop)] for j in range(nop)]
        for i in range(nop): # loop over first operator
            for j in range(nop): # loop over second operator
                A = ops[i] # first operator
                B = ops[j] # second operator
                es,chisi = chiAB(h,mode="matrix",A=A,B=B,
                                **kwargs) # non-interacting response
                chis[i][j] = chisi # store in the list
        # now make it a block matrix, and reshpae accordingly
        chis_tmp = np.array(chis) # convert to array
        chis = [] # empty list
        for i in range(len(es)): # loop over energies
            chi = chis_tmp[:,:,i,:,:] # get this one
            chi = [[chi[i,j,:,:] for i in range(nop)] for j in range(nop)]
            chis.append(np.bmat(chi)) # store
    elif mode_rpa=="vectorized": # all at once
        # operators as matrices
        from .. import operators
        pAs = [] # empty list
        pBs = [] # empty list
        projs = [operators.index(h,n=[i]) for i in range(len(h.geometry.r))]
        for i in range(nop): # loop over first operator
            A = ops[i] # first operator
            B = ops[i] # second operator
            A = algebra.todense(h.get_operator(A).get_matrix())
            B = algebra.todense(h.get_operator(B).get_matrix())
            for pi in projs: # products
                pAs.append(pi@A)
                pBs.append(pi@B)
        es,chis = chiAB(h,mode="matrix",pAs=pAs,pBs=pBs,
                                **kwargs) # non-interacting response
    else: raise
    iden = np.identity(chis[0].shape[0],dtype=np.complex128) # identity
    if V is not None: # finite interaction, RPA summation
        chis_rpa = [chi@algebra.inv(iden - V@chi) for chi in chis]
    else: chis_rpa = chis
    return es,np.array(chis_rpa)




def chi_AB_RPA_scf(scf):
    """Return the RPA response function for an SCF object"""
    if len(scf.v)==1: # just the onsite term
        return chi_AB_RPA(scf.hamiltonian,scf.v[(0,0,0)])
    else: raise # not implemented


def spinchi_pm_RPA(h,U=0.,v=[0.,0.,1.],**kwargs):
    """Compute the spin RPA response for a hamiltonian.
     - v is the chosen quantization axis of the ladder operators
     - U is the Hubbard interaction"""
     # v needs to be implemented
    sx = h.get_operator("sx") # spin operator, eigen +-1
    sy = h.get_operator("sy") # spin operator, eigen +-1
    sz = h.get_operator("sz") # spin operator, eigen +-1
    v = np.array(v) # convert to array
    sp = (sx + 1j*sy)/2. # ladder operator
    sm = (sx - 1j*sy)/2. # ladder operator
    from ..chi import chiAB # get response function
    es,chis = chiAB(h,A=sp,B=sm,mode="matrix",**kwargs) # non-interacting response functions
    iden = np.identity(chis[0].shape[0],dtype=np.complex128) # identity
    # this factor 1/2 should be here
    chisrpa = [chi@algebra.inv(iden + U/2.*chi) for chi in chis] # RPA summation
    return es,np.array(chisrpa) # return energies and RPA response function




