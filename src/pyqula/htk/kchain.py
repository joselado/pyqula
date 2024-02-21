import numpy as np


#def kchain_dense_old(h,k):
#  """ Return the kchain Hamiltonian """
#  if h.dimensionality != 2: raise
#  if h.is_multicell: h = h.get_no_multicell() # redefine
#  tky = h.ty*np.exp(1j*np.pi*2.*k)
#  tkxy = h.txy*np.exp(1j*np.pi*2.*k)
#  tkxmy = h.txmy*np.exp(-1j*np.pi*2.*k)  # notice the minus sign !!!!
#  # chain in the x direction
#  ons = h.intra + tky + np.conjugate(tky).T  # intra of k dependent chain
#  hop = h.tx + tkxy + tkxmy  # hopping of k-dependent chain
#  return (ons,hop)


def kchain(h,**kwargs):
    """Return the onsite, t1 and t2"""
    if not h.is_multicell: h = h.get_multicell()
    # make a check that only NN matters
    if detect_longest_hopping(h)==1:
        hnn = h.get_no_multicell() # no multicell Hamiltonian
        return kchain_NN(h,**kwargs) # only NN coupling
    elif detect_longest_hopping(h)==2:
        print("WARNING, NNN in kchain")
        return kchain_NNN(h,**kwargs) # include NNN
    else: raise



def kchain_NNN(h,k=[0.,0.,0.]):
    """Return the onsite, t1 and t2"""
    if not h.is_multicell: h = h.get_multicell()
    dim = h.dimensionality # dimensionality
    if dim==1: # 1D
        t1 = h.intra*0.
        t2 = h.intra*0.
        for t in h.hopping:
            if t.dir[0]==1: t1 = t.m 
            if t.dir[0]==2: t2 = t.m 
        return h.intra,t1,t2
    elif dim>1: # 2D or 3D
        intra = np.zeros(h.intra.shape,dtype=np.complex_) # zero amtrix
        inter1 = np.zeros(h.intra.shape,dtype=np.complex_) # zero amtrix
        inter2 = np.zeros(h.intra.shape,dtype=np.complex_) # zero amtrix
        intra = h.intra # initialize
        for t in h.hopping: # loop over hoppings
            tk = t.m * h.geometry.bloch_phase(t.dir,k) # k hopping
            if t.dir[dim-1]==0: intra = intra + tk # add contribution 
            if t.dir[dim-1]==1: inter1 = inter1 + tk # add contribution 
            if t.dir[dim-1]==2: inter2 = inter2 + tk # add contribution 
        return intra,inter1,inter2
    else: raise


def kchain_NN(h,k=[0.,0.,0.]):
    """Return the onsite and hopping for a particular k"""
    if not h.is_multicell: h = h.get_multicell()
    # make a check that only NN matters
    dim = h.dimensionality # dimensionality
    if dim==1: # 1D
        for t in h.hopping:
            if t.dir[0]==1: return h.intra,t.m
        raise
    elif dim>1: # 2D or 3D
      intra = np.zeros(h.intra.shape) # zero amtrix
      inter = np.zeros(h.intra.shape) # zero amtrix
      intra = h.intra # initialize
      for t in h.hopping: # loop over hoppings
        tk = t.m * h.geometry.bloch_phase(t.dir,k) # k hopping
        if t.dir[dim-1]==0: intra = intra + tk # add contribution 
        if t.dir[dim-1]==1: inter = inter + tk # add contribution 
      return intra,inter
    else: raise


def detect_longest_hopping(h,tol=1e-7):
    h = h.get_multicell() # multicell Hamiltonian
    out = 0 # initialize
    for t in h.hopping: # loop over hoppings
        if np.max(np.abs(t.m))>tol: # if bigger than the tolerance
            nn = np.max(np.abs(t.dir))
            if nn>out: out = nn # overwrite
    return out




def kchain_LR(h,k=[0.,0.,0.]):
    """Return the onsite, t1 and t2"""
    if not h.is_multicell: h = h.get_multicell()
    dim = h.dimensionality # dimensionality
    zero = h.intra*0j # initialize
    numt = detect_longest_hopping(h)
    hops = [zero.copy() for i in range(numt+1)] # empty list with hoppings
    hops[0] = h.intra.copy() # store this one
    if dim==1: # 1D
        for t in h.hopping:
            if t.dir[0]>0: # positive ones
                hops[t.dir[0]] = t.m.copy() # store this hopping
        return hops
    elif dim>1: # 2D or 3D
        for t in h.hopping: # loop over hoppings
            tk = t.m * h.geometry.bloch_phase(t.dir,k) # k hopping
#            if t.dir[dim-1]==0: intra = intra + tk # add contribution 
            if t.dir[dim-1]>=0: # positive ones and intra
                hops[t.dir[dim-1]] += tk # add this hopping
        return hops
    else: raise








