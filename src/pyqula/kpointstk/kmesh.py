import numpy as np




def kmesh(dimensionality,nk=10,nsuper=1,
        endpoint=False):
    """Return a mesh of k-points for a certain dimensionality. nk is either
    one number, used in every direction, or one number per direction, as
    a list, a tuple or a numpy array"""
    kp = []
    if dimensionality==0: return [[0.,0.,0.]]
    nk = nk_per_direction(nk,dimensionality)
    if all(n==1 for n in nk): return [[0.,0.,0.]]
    if dimensionality==1:
        for k1 in np.linspace(0.,nsuper,nk[0],endpoint=endpoint):
          kp.append([k1,0.,0.]) # store
    elif dimensionality==2:
        kp = kmesh2d(nk,nsuper,endpoint=endpoint) # get 2D kmesh
    elif dimensionality==3:
        for k1 in np.linspace(0.,nsuper,nk[0],endpoint=endpoint):
          for k2 in np.linspace(0.,nsuper,nk[1],endpoint=endpoint):
            for k3 in np.linspace(0.,nsuper,nk[2],endpoint=endpoint):
              kp.append([k1,k2,k3]) # store
    else:
        raise ValueError("a k-mesh needs a dimensionality between 0 and 3")
    kp = [np.array(k) for k in kp] # to array
    return np.array(kp)


def nk_per_direction(nk,dimensionality):
    """Return nk as a list with one entry per direction. A scalar, a Python
    or numpy integer or a 0-d array, is used for every direction; a
    sequence or a 1-d array gives one entry per direction, and a single
    entry is used for all of them. Going through np.asarray is what makes
    a numpy array behave like a list: comparing an array with a number
    gives an array, whose truth value is ambiguous"""
    nks = np.asarray(nk)
    if nks.ndim==0: return [nks.item()]*dimensionality
    nks = nks.ravel().tolist()
    if len(nks)==1: return nks*dimensionality
    if len(nks)<dimensionality:
        raise ValueError("a k-mesh in "+str(dimensionality)+" dimensions "
                "needs nk as one number or one per direction, got "
                +str(nk))
    return nks[:dimensionality]



from numba import jit

#@jit(nopython=True)
## there is some problem with endpoint in numba
# this should be probably fixed for compatibility
def kmesh2d(nk,nsuper,endpoint=False):
    nkt = nk[0]*nk[1] # total number of kpoints
    kp = np.zeros((nkt,3),dtype=np.float64) # kpoints
    ik = 0
    for k1 in np.linspace(0.,nsuper,nk[0],endpoint=endpoint):
        for k2 in np.linspace(0.,nsuper,nk[1],endpoint=endpoint):
            kp[ik] = np.array((k1,k2,0.)) # store
            ik += 1 # increase counter
    return kp



