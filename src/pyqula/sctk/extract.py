import numpy as np
from ..superconductivity import get_eh_sector
from ..superconductivity import build_nambu_matrix
from ..multihopping import MultiHopping
from .. import algebra

def extract_anomalous_dict(dd):
    """Given a dictionary, extract the anomalous part"""
    out = dict()
    for key in dd:
        d = dd[key] # get this patrix
        m01 = get_eh_sector(d,i=0,j=1)
        m10 = get_eh_sector(d,i=1,j=0)
        m = build_nambu_matrix(m01*0.0,c12=m01,c21=m10) # build matrix
        out[key] = m
    return out # return dictionary


def extract_normal_dict(dd):
    """Given a dictionary, extract the anomalous part"""
    out = dict()
    for key in dd:
        d = dd[key] # get this patrix
        m00 = get_eh_sector(d,i=0,j=0)
        m = build_nambu_matrix(m00) # build matrix
        out[key] = m
    return out # return dictionary




def get_anomalous_hamiltonian(self):
    """Return anomalous part of a Hamiltonian"""
    h0 = self.copy() ; h0.remove_nambu() ; h0.setup_nambu_spinor()
    h = self - h0
    return h
#    self = self.copy() # copy Hamiltonian
#    self.turn_nambu() # setup electron-hole if not present
#    dd = self.get_multihopping().get_dict() # return the dictionary
#    dd = extract_anomalous_dict(dd)
#    self.set_multihopping(MultiHopping(dd))
#    return self


def get_singlet_hamiltonian(self):
    self = self.copy()
    dd = self.get_multihopping().get_dict() # return the dictionary
    dd = extract_singlet_dict(dd)
    self.set_multihopping(MultiHopping(dd))
    return self


def get_triplet_hamiltonian(self):
    h = get_anomalous_hamiltonian(self)
    return h - get_singlet_hamiltonian(h)



def extract_pairing(m):
  """Extract the pairing from a matrix, assuming it has the Nambu form"""
  nr = m.shape[0]//4 # number of positions
  uu = np.array(np.zeros((nr,nr),dtype=np.complex)) # zero matrix
  dd = np.array(np.zeros((nr,nr),dtype=np.complex)) # zero matrix
  ud = np.array(np.zeros((nr,nr),dtype=np.complex)) # zero matrix
  for i in range(nr): # loop over positions
    for j in range(nr): # loop over positions
        ud[i,j] = m[4*i,4*j+2]
        dd[i,j] = m[4*i+1,4*j+2]
        uu[i,j] = m[4*i,4*j+3]
  return (uu,dd,ud) # return the three matrices



def extract_triplet_pairing(m):
  """Extract the pairing from a matrix, assuming it has the Nambu form"""
  m = algebra.todense(m) # dense matrix
  nr = m.shape[0]//4 # number of positions
  uu = np.array(np.zeros((nr,nr),dtype=np.complex)) # zero matrix
  dd = np.array(np.zeros((nr,nr),dtype=np.complex)) # zero matrix
  ud = np.array(np.zeros((nr,nr),dtype=np.complex)) # zero matrix
  for i in range(nr): # loop over positions
    for j in range(nr): # loop over positions
        ud[i,j] = (m[4*i,4*j+2] - np.conjugate(m[4*j+3,4*i+1]))/2.
        dd[i,j] = m[4*i+1,4*j+2]
        uu[i,j] = m[4*i,4*j+3]
  return (uu,dd,ud) # return the three matrices


def extract_singlet_dict(dd):
    """Given a dictionary, extract the anomalous singlet part"""
    out = dict()
    for key in dd:
        d = dd[key] # get this matrix
        nr = d.shape[0]//4 # number of positions
        m = np.zeros(d.shape,dtype=np.complex) # initialize
        m0 = dd[key]
        key2 = (-key[0],-key[1],-key[2]) 
        m1 = dd[key2]
        for i in range(nr): # loop over positions
            for j in range(nr): # loop over positions
                m[4*i,4*j+2] = (m0[4*i,4*j+2]+np.conjugate(m1[4*j+3,4*i+1]))/2.
                m[4*j+3,4*i+1] = (m0[4*j+3,4*i+1]+np.conjugate(m1[4*i,4*j+2]))/2.
                m[4*i+2,4*j] = (m0[4*i+2,4*j]+np.conjugate(m1[4*j+1,4*i+3]))/2.
                m[4*j+1,4*i+3] = (m0[4*j+1,4*i+3]+np.conjugate(m1[4*i+2,4*j]))/2.
        out[key] = m
    return out # return dictionary


def extract_triplet_dict(dd):
    dd = extract_anomalous_dict(dd) # anomalous part
    ds = extract_singlet_dict(dd) # singlet part
    from ..multihopping import MultiHopping
    return (MultiHopping(dd) - MultiHopping(ds)).get_dict()



def extract_singlet_pairing(m):
  """Extract the pairing from a matrix, assuming it has the Nambu form"""
  nr = m.shape[0]//4 # number of positions
  ud = np.array(np.zeros((nr,nr),dtype=np.complex)) # zero matrix
  for i in range(nr): # loop over positions
    for j in range(nr): # loop over positions
        ud[i,j] = (m[4*i,4*j+2] + np.conjugate(m[4*j+3,4*i+1]))/2.
  return ud



def extract_custom_pairing(m,mode="all"):
    """Given a matrix, extract the pairing matrix according to some rule"""
    if mode=="singlet": # singlet, with sign
        m = extract_singlet_pairing(m) # matrix with pairings 
        return m
    elif mode=="triplet": # triplet, summed over
        ms = extract_triplet_pairing(m) # matrix with pairings 
        m = np.sum(np.abs(np.array(ms))**2,axis=0)
        return m
    elif mode=="all": # compute all in absolute value
        ms = extract_triplet_pairing(m) # matrix with pairings (3 of them)
        mt = np.sum(np.abs(np.array(ms))**2,axis=0)
        m = extract_singlet_pairing(m) # matrix with pairings 
        m = mt + np.abs(np.array(m))**2 # singlet plus triplet
        return m
    elif mode=="both": # singlet and triplet with interference effects
        if m.shape[0]==4: # this is a quick fix for singlet site models
            m = m@np.conjugate(m.T)
            return np.array([[np.trace(m)]])
        else: raise
    else: raise



def extract_pairing_kmap(h,write=False,i=None,j=None,mode="all",**kwargs):
    """Extract the pairing in reciprocal space"""
    if not h.has_eh: raise # not implemented
    h = get_anomalous_hamiltonian(h)
    if j is None: j = i # same site is the default
    fk = h.get_hk_gen() # Bloch Hamiltonian generator
    def f0(k):
        m = fk(k) # full k-dependent Hamiltonian
        m = extract_custom_pairing(m,mode=mode)
        if i is None: return np.sqrt(np.trace(m@np.conjugate(m.T)))
        else: return m[i,j] # return pairing
    dref = f0(np.random.random(3)) ; dref = dref/np.abs(dref) # reference
    fr = lambda k: (f0(k)/dref).real # reference
    fi = lambda k: (f0(k)/dref).imag # reference
    from .. import spectrum
    (ks,dsr) = spectrum.reciprocal_map(h,fr,write=write,**kwargs)
    (ks,dsi) = spectrum.reciprocal_map(h,fi,write=write,**kwargs)
    return ks[:,0],ks[:,1],dsr+1j*dsi






