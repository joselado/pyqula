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
    """Turn a Hamiltonian into a Nambu Hamiltonian"""
    self = self.copy() # copy Hamiltonian
    self.turn_nambu() # setup electron-hole if not present
    dd = self.get_multihopping().get_dict() # return the dictionary
    dd = extract_anomalous_dict(dd)
    self.set_multihopping(MultiHopping(dd))
    return self


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




