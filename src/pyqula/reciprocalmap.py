import numpy as np

def reciprocal_evaluation(h,f,nk=50,nsuper=1,reciprocal=True):
    """Compute a reciprocal map"""
    if h.dimensionality!=2: raise  # continue if two dimensional
    kxs = np.linspace(-nsuper,nsuper,nk,endpoint=True)  # generate kx
    kys = np.linspace(-nsuper,nsuper,nk,endpoint=True)  # generate ky
    if reciprocal: fR = h.geometry.get_k2K_generator() # get matrix
    else:  fR = lambda x: x # get identity
    # setup the operator
    rs = [] # empty list
    for x in kxs:
      for y in kys:
        rs.append([x,y,0.]) # store
    def getf(r): 
        return f(fR(r)) # function to compute
    rs = np.array(rs) # transform into array
    from . import parallel
    kx = rs[:,0] # x coordinate
    ky = rs[:,1] # y coordinate
    out = [getf(r) for r in rs]
#    out = parallel.pcall(getf,rs) # compute all
    out = np.array(out) # transform into an array
    return kx,ky,out




def sc_kmap(h,fh,**kwargs):
    """Compute the reciprocal space map of spin singlet SC"""
    h = h.copy()
    h = fh(h)
    hk = h.get_hk_gen() # get Bloch Hamiltonian generator
    def f(k):
        m = hk(k)
        m = m@m
        return np.trace(m).real # return the trace
    kx,ky,out = reciprocal_evaluation(h,f,**kwargs)
    np.savetxt("RECIPROCAL_MAP.OUT",np.array([kx,ky,out]).T)

def singlet_map(h,**kwargs):
    from .sctk.extract import get_singlet_hamiltonian as fh
    return sc_kmap(h,fh,**kwargs)


def triplet_map(h,**kwargs):
    from .sctk.extract import get_triplet_hamiltonian as fh
    return sc_kmap(h,fh,**kwargs)

