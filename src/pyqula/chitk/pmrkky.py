
from __future__ import print_function
import scipy.linalg as lg
import numpy as np
from .. import algebra


def explicit_rkky(h,ri=None,rj=None,nk=10,dj=1e-1):
    """Compute the RKKY interaction explicitly by adding
    local exchange fields"""
    if ri is None: raise
    if rj is None: raise
    h0 = h.copy() # copy Hamiltonian
    h0.turn_spinful() # spinfull Hamiltonian
    filling = h0.get_filling(nk=nk) # get the filling of the Hamiltonian
    hfe = h0.copy() # first Hamiltonian
    haf = h0.copy() # second Hamiltonian
    from ..potentials import delta
    v1 = delta(ri,v0=np.array([0.,0.,1.])*dj) # local exchange
    v2 = delta(rj,v0=np.array([0.,0.,1.])*dj) # local exchange
    v3 = delta(rj,v0=np.array([0.,0.,-1.])*dj) # local exchange
    hfe.add_exchange(v1) # add exchange
    hfe.add_exchange(v2) # add exchange
    haf.add_exchange(v1) # add exchange
    haf.add_exchange(v3) # add exchange
    # correction due to a potential shift of Fermi energy
    fermife = hfe.get_fermi4filling(filling,nk=nk) # FE
    fermiaf = haf.get_fermi4filling(filling,nk=nk) # FE
#    hfe.shift_fermi(-fermife)
#    haf.shift_fermi(-fermiaf)
    efe = hfe.get_total_energy(nk=nk,fermi=fermife) # total energy
    eaf = haf.get_total_energy(nk=nk,fermi=fermiaf) # total energy
#    print("Fermi shift",fermife-fermiaf)
    de = efe - eaf #+ (-fermife+fermiaf) # energy difference
    return -1./4.*de/dj**2 # return RKKY


# Poor man RKKY routines, using brute force energetics
# mainly for 0d, yet it can apply to 2d with PBC

def rkky_atom(hin,delta=0.001,i=None,filling=0.5):  
  if h.dimensionality != 0: raise # only for 0d
  if i is None: raise # default value
  else: r0 = hin.geometry.r[i]
  no = hin.intra.shape[0]
  # unperturbed hamiltonian
  (evals,evecs) = algebra.eigh(hin.intra)
  den0 = evecs[0]*0.0 # initializa density
  for i in range(no/2): # loop over sites
    den0 += evecs[i]*np.conjugate(evecs[i]) # density
  def denshift(delta2):
    nfill = int(round(no*filling)) # filling
    def fermi(r):
      dr = r0 - np.array(r)
      dr = dr.dot(dr)
      if dr<0.1: return delta2
  #    if x<0.0 and y>0.0: return delta
      else: return 0.0
    h = hin.copy()
    # now perturb the hamiltonian
    h.shift_fermi(fermi)
    intra = h.intra # intraterm
    (evals,evecs) = algebra.eigh(intra)
    evecs = evecs.transpose()
    eper = 0.0 # perturbed energy
    den = evecs[0]*0.0 # initializa density
    for i in range(nfill): # loop over sites
      den += evecs[i]*np.conjugate(evecs[i]) # density
    return den.real
  denup,dendn = denshift(-delta),denshift(delta) # calculate density up and dn
  den = denup - dendn
  return den # return row of correlation matrix




def rkky0d(h,check=False):
  """Calculate RKKY interaction for a 0d system"""
  if h.dimensionality != 0: raise # only for 0d
  if h.has_spin: raise # only for spinless
  if h.has_eh: raise # only for electrons
  m = [] #empty matrix
  for i in range(h.intra.shape[0]): # loop over sites
    row = rkky_atom_v1(h,i=i)
    m.append(row)
  m = np.array(m)
  diff = np.sum(np.abs(m - np.transpose(m)))
  if check: # check that it is symmetric
    if diff>0.01: raise
    print("RKKY matrix is symmetric")
  m = (m + np.transpose(m))/2.0  # symmetrize
  return m # return correlation matrix




def rkky_atom_v1(hin,delta=0.001,i=None,filling=0.5):  
  if hin.dimensionality != 0: raise # only for 0d
  if i is None: raise # default value
  else: r0 = hin.geometry.r[i]
  no = hin.intra.shape[0]
  def denshift(delta2):
    def fermi(r):
      dr = r0 - np.array(r)
      dr = dr.dot(dr)
      if dr<0.1: return delta2
  #    if x<0.0 and y>0.0: return delta
      else: return 0.0
    h = hin.copy()
    # now perturb the hamiltonian
    h.shift_fermi(fermi)
    intra = h.intra # intraterm
    (evals,evecs) = algebra.eigh(intra)
    evecs = evecs.transpose()
    return evals,evecs
  # get eigenvalues and eigenvectors for up/down
  eup,vup = denshift(-delta)
  edn,vdn = denshift(delta)
  # get fermi
  nfill = int(round(2*no*filling)) # filling
  es = sorted(np.concatenate((eup,edn))) # list of energies
  fermi = (es[nfill] + es[nfill-1])/2.0 # fermi energy
  mag = (vup[0]*0.).real # initialize
  for i in range(len(eup)):
    if eup[i]<fermi:
      dup = vup[i]*np.conjugate(vup[i]) # density up
      mag += dup.real # magnetization
    if edn[i]<fermi:
      ddn = vdn[i]*np.conjugate(vdn[i]) # density down
      mag -= ddn.real # magnetization
  return mag/delta # return row of correlation matrix



def write_rkky(h,output_file="RKKY.OUT",write_all=True,check=False):
  """ Writes in file the RKKY interaction"""
  m = rkky0d(h,check=check) # get the matrix
  evals,evecs = lg.eigh(m) # eigenvalues and eigenvectors
  evecs = np.transpose(evecs) # transpose eigenvectors
#  den = (evecs[-1]*np.conjugate(evecs[-1])).real
  den = evecs[-1].real
  ns = h.intra.shape[0] # number of sites
  fo = open(output_file,"w") # open file
  for i in range(ns):
    fo.write(str(h.geometry.x[i])+"  ")
    fo.write(str(h.geometry.y[i])+"  ")
    fo.write(str(den[i])+"\n")
  fo.close()
  if write_all:
    for j in range(len(evals)):
      name = "EIGEN_RKKY_"+str(len(evals)-j-1)+"_value_"+str(evals[j])+".OUT" # name of the file
      fo = open(name,"w")
      den = evecs[j].real
      for i in range(ns): # loop over atoms
        fo.write(str(h.geometry.x[i])+"  ")
        fo.write(str(h.geometry.y[i])+"  ")
        fo.write(str(den[i])+"\n")
      fo.close()
  # now write eigenvalues of RKKY
  fo = open("EIGENVALUES_RKKY.OUT","w")
  for i in range(len(evals)):
    fo.write(str(i)+"  ")
    fo.write(str(evals[i])+"\n")
  fo.close()
  return den

