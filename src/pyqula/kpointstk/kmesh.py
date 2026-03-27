import numpy as np
from ..checkclass import number2array




def kmesh(dimensionality,nk=10,nsuper=1,
        endpoint=False):
    """Return a mesh of k-points for a certain dimensionality"""
    kp = []
    if nk==1 or dimensionality==0: return [[0.,0.,0.]]
    if dimensionality==1:
        for k1 in np.linspace(0.,nsuper,nk,endpoint=endpoint):
          kp.append([k1,0.,0.]) # store
    elif dimensionality==2:
        nk = number2array(nk) # return an array
        kp = kmesh2d(nk,nsuper) # get 2D kmesh
    elif dimensionality==3:
        nk = number2array(nk) # return an array
        for k1 in np.linspace(0.,nsuper,nk[0],endpoint=endpoint):
          for k2 in np.linspace(0.,nsuper,nk[1],endpoint=endpoint):
            for k3 in np.linspace(0.,nsuper,nk[2],endpoint=endpoint):
              kp.append([k1,k2,k3]) # store
    else: raise
    kp = [np.array(k) for k in kp] # to array
    return np.array(kp)



from numba import jit

#@jit(nopython=True)
## there is some problem with endpoint in numba
# this should be probably fixed for compatibility
def kmesh2d(nk,nsuper):
    nkt = nk[0]*nk[1] # total number of kpoints
    kp = np.zeros((nkt,3),dtype=np.float64) # kpoints
    ik = 0
    for k1 in np.linspace(0.,nsuper,nk[0],endpoint=False):
        for k2 in np.linspace(0.,nsuper,nk[1],endpoint=False):
            kp[ik] = np.array((k1,k2,0.)) # store
            ik += 1 # increase counter
    return kp



