import numpy as np
from ..multihopping import MultiHopping
from .. import algebra

# superconductivity.py imports several names from this module, so importing
# get_eh_sector/build_nambu_matrix at the top level here would make the two
# modules unimportable depending on which one is reached first; they are
# imported inside the two functions that use them instead

def extract_anomalous_dict(dd):
    """Given a dictionary, extract the anomalous part"""
    from ..superconductivity import get_eh_sector,build_nambu_matrix
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
    from ..superconductivity import get_eh_sector,build_nambu_matrix
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


def get_singlet_hamiltonian(self):
    """Return only the anomalous spin singlet superconducting state"""
    self = self.copy()
    dd = self.get_multihopping().get_dict() # return the dictionary
    dd = extract_singlet_dict(dd)
    self.set_multihopping(MultiHopping(dd))
    return self


def get_triplet_hamiltonian(self):
    """Return only the anomalous spin singlet superconducting state"""
    h = get_anomalous_hamiltonian(self)
    return h - get_singlet_hamiltonian(h)



def extract_pairing(m):
  """Extract the pairing from a matrix, assuming it has the Nambu form"""
  # the four spin x electron-hole components of a site are consecutive, so
  # every pairing block is a constant-stride slice, ud[i,j] = m[4*i,4*j+2]
  m = algebra.todense(m) # dense matrix
  ud = np.array(m[0::4,2::4],dtype=np.complex128)
  dd = np.array(m[1::4,2::4],dtype=np.complex128)
  uu = np.array(m[0::4,3::4],dtype=np.complex128)
  return (uu,dd,ud) # return the three matrices



def extract_triplet_pairing(m):
  """Extract the pairing from a matrix, assuming it has the Nambu form"""
  m = algebra.todense(m) # dense matrix
  # strided slices, the transpose implements the (4*j+3,4*i+1) index swap
  ud = (np.array(m[0::4,2::4]) - np.conjugate(np.array(m[3::4,1::4]).T))/2.
  dd = np.array(m[1::4,2::4],dtype=np.complex128)
  uu = np.array(m[0::4,3::4],dtype=np.complex128)
  return (uu,dd,ud) # return the three matrices


def extract_singlet_dict(dd):
    """Given a dictionary, extract the anomalous singlet part"""
    out = dict()
    for key in dd:
        d = dd[key] # get this matrix
        m = np.zeros(d.shape,dtype=np.complex128) # initialize
        key2 = (-key[0],-key[1],-key[2]) 
        m0 = algebra.todense(dd[key]) # dense matrix
        m1 = algebra.todense(dd[key2]) # dense matrix
        # the four blocks are strided slices, and the index swap between the
        # two matrices of a pair is a transpose
        m[0::4,2::4] = (np.array(m0[0::4,2::4])
                       +np.conjugate(np.array(m1[3::4,1::4]).T))/2.
        m[3::4,1::4] = (np.array(m0[3::4,1::4])
                       +np.conjugate(np.array(m1[0::4,2::4]).T))/2.
        m[2::4,0::4] = (np.array(m0[2::4,0::4])
                       +np.conjugate(np.array(m1[1::4,3::4]).T))/2.
        m[1::4,3::4] = (np.array(m0[1::4,3::4])
                       +np.conjugate(np.array(m1[2::4,0::4]).T))/2.
        out[key] = m
    return out # return dictionary


def extract_triplet_dict(dd):
    dd = extract_anomalous_dict(dd) # anomalous part
    ds = extract_singlet_dict(dd) # singlet part
    from ..multihopping import MultiHopping
    return (MultiHopping(dd) - MultiHopping(ds)).get_dict()



def extract_singlet_pairing(m):
  """Extract the pairing from a matrix, assuming it has the Nambu form"""
  m = algebra.todense(m) # dense matrix
  # strided slices, the transpose implements the (4*j+3,4*i+1) index swap
  ud = (np.array(m[0::4,2::4]) + np.conjugate(np.array(m[3::4,1::4]).T))/2.
  return ud




def extract_singlet_hamiltonian(h):
    """Given a Hamiltonian, return only the superconducting singlet term"""
    h = h.copy() # copy Hamiltonian
    dd = h.get_dict() # extract dictionary
    dd = extract_singlet_dict(dd) # extract the singlet
    from ..multicell import set_dictionary
    h = set_dictionary(h,dd) # set the dictionary
    return h # return the singlet Hamiltonian


def extract_triplet_hamiltonian(h):
    """Given a Hamiltonian, return only the superconducting triplet term"""
    h = get_anomalous_hamiltonian(h)
    return h - extract_singlet_hamiltonian(h) # return the triplet



def extract_custom_pairing(m,mode="all"):
    """Given a matrix, extract the pairing matrix according to some rule"""
    raise NotImplementedError("extract_custom_pairing is not implemented; use "
            "extract_absolute_pairing or extract_pairing_kmap instead")
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
        if m.shape[0]==4: # this is a quick fix for single site models
            m = m@np.conjugate(m.T)
            return np.array([[np.trace(m)]])
        else:
            raise NotImplementedError("the 'both' mode is only implemented "
                    "for single-site models")
    else:
        raise ValueError("unknown mode; the accepted ones are 'all', "
                "'singlet', 'triplet' and 'both'")



def extract_pairing_kmap(h,write=False,i=None,j=None,mode="all",**kwargs):
    """Extract the pairing in reciprocal space"""
    if not h.has_eh:
        raise ValueError("the pairing k-map needs a Nambu Hamiltonian; call "
                "h.setup_nambu_spinor() first")
    h = get_anomalous_hamiltonian(h)
    if j is None: j = i # same site is the default
    if mode=="all": pass # do nothing 
    elif mode=="singlet": h = extract_singlet_hamiltonian(h) # singlet
    elif mode=="triplet": h = extract_triplet_hamiltonian(h) # triplet
    else:
        raise ValueError("unknown mode; the accepted ones are 'all', "
                "'singlet' and 'triplet'")
    fk = h.get_hk_gen() # Bloch Hamiltonian generator
    def f0(k):
        m = fk(k) # full k-dependent Hamiltonian
        if i is None: return np.trace(m@np.conjugate(m.T))/m.shape[0]
        else:
            raise NotImplementedError("a single pairing matrix element is not "
                    "implemented, only the trace over all of them")
        # return m[i,j] # return pairing
    from .. import spectrum
    (ks,ds) = spectrum.reciprocal_map(h,f0,write=write,**kwargs)
    return ks[:,0],ks[:,1],ds
#    dref = f0(np.random.random(3)) ; dref = dref/np.abs(dref) # reference
#    fr = lambda k: (f0(k)/dref).real # reference
#    fi = lambda k: (f0(k)/dref).imag # reference
#    from .. import spectrum
#    (ks,dsr) = spectrum.reciprocal_map(h,fr,write=write,**kwargs)
#    (ks,dsi) = spectrum.reciprocal_map(h,fi,write=write,**kwargs)
#    return ks[:,0],ks[:,1],dsr+1j*dsi


def extract_absolute_pairing(h,mode="singlet",**kwargs):
    """Extract the absolute value of the SC order in the BZ"""
    h = get_anomalous_hamiltonian(h)
    if mode=="all" or mode=="both": pass # do nothing 
    elif mode=="singlet": h = extract_singlet_hamiltonian(h) # singlet
    elif mode=="triplet": h = extract_triplet_hamiltonian(h) # triplet
    else: # do nothing
        raise ValueError("unknown mode; the accepted ones are 'all', 'both', "
                "'singlet' and 'triplet'")
    fk = h.get_hk_gen() # Bloch Hamiltonian generator
    from ..klist import kmesh
    ks = kmesh(h.dimensionality,**kwargs) # kpoints
    def f(k):
        m = fk(k) # return Bloch Hamiltonian
        return np.trace(m@m)/m.shape[0]
    return np.sqrt(np.mean([f(k) for k in ks])) # return mean value




def extract_absolute_spatial_pairing(h,mode="singlet",**kwargs):
    """Extract the absolute value of the SC order in the BZ"""
    h = get_anomalous_hamiltonian(h) # overwrite Hamiltonian
    if mode=="all" or mode=="both": pass # do nothing
    elif mode=="singlet": h = extract_singlet_hamiltonian(h) # singlet
    elif mode=="triplet": h = extract_triplet_hamiltonian(h) # triplet
    else: # do nothing
        raise ValueError("unknown mode; the accepted ones are 'all', 'both', "
                "'singlet' and 'triplet'")
    fk = h.get_hk_gen() # Bloch Hamiltonian generator
    from ..klist import kmesh
    ks = kmesh(h.dimensionality,**kwargs) # kpoints
    def f(k):
        m = fk(k) # return Bloch Hamiltonian
        m2 = m@m # compute H**2
        out = np.diag(m2) # return the diagonal
        return out
    out = np.mean([f(k) for k in ks],axis=0) # return mean value
    from ..increase_hilbert import full2profile
    out = full2profile(h,out) # resum
    # full2profile has added the spin x electron-hole components of each
    # site, and every one of them contributes |Delta_i|**2, so dividing by
    # the number of components per site leaves |Delta_i|
    nc = h.intra.shape[0]//len(h.geometry.r) # components per site
    return np.sqrt(out.real/nc) # return the spatial profile



