import os
import numpy as np
from ..scftypes import scfclass
from ..scftypes import get_fermi_energy
from ..scftypes import get_occupied_states
from scipy.sparse import csc_matrix
import time
from ..scftypes import directional_mean_field
from .. import limits
from .. import inout


try:
  from .. import correlatorsf90
except: pass



mf_file = "MF.pkl" # mean field file

def hubbardscf(h,g=1.0,nkp = 100,filling=0.5,mag=None,mix=0.9,
                  maxerror=1e-05,silent=False,mf=None,
                  T=None,collinear=False,fermi_shift=0.0,
                  maxite=1000,save=False,nk=None,U=None):
  """ Solve a selfconsistent Hubbard mean field"""
  # alternative input variables
  if nk is not None: nkp = nk
  if U is not None: g = U
  #############################
  mix = 1. - mix
  U = g # redefine
  if not h.has_spin: raise
  os.system("rm -f STOP") # remove stop file
  from scipy.linalg import eigh
  nat = h.intra.shape[0]//2 # number of atoms
  htmp = h.copy()  # copy hamiltonian
  htmp.turn_dense() # turn into a dense Hamiltonian
  # generate the necessary list of correlators
  if mf is None: # generate initial mean field
    try:  
        old_mf = inout.load(mf_file) # load the file
        print("Mean field read from file")
    except: # generate mean field
      if mag is None: mag = np.random.random((nat,3))
# get mean field matrix
      old_mf = selective_U_matrix(U,directional_mean_field(mag))
  else: old_mf = mf # use guess
  # get the pairs for the correlators
  ndim = h.intra.shape[0] # dimension
  totkp = nkp**(h.dimensionality) # total number of kpoints
  file_etot = open("SCF_ENERGY.OUT","w")
  file_error = open("SCF_ERROR.OUT","w")
  ite = 0 # iteration counter
  scf = scfclass(h) # create scf class
  while True: # infinite loop
    ite += 1 # increase counter
    htmp.intra = h.intra + old_mf # add mean field 
    t1 = time.time()
# get eigenvectors
    eigvals,eigvecs,kvectors = htmp.get_eigenvectors(nk=nkp,kpoints=True)
    eigvecs = np.conjugate(eigvecs)
# fermi energy
    t2 = time.time()
    fermi = get_fermi_energy(eigvals,filling,fermi_shift=fermi_shift)
# occupied states
    eoccs,voccs,koccs = get_occupied_states(eigvals,eigvecs,kvectors,fermi,
            smearing=T)
# mean field
    mf,edc,charge,mag = magnetic_mean_field(voccs,U,collinear=collinear,
                                totkp=totkp)
    t3 = time.time()
    error = np.max(np.abs(old_mf-mf)) # absolute difference
    # total energy
    print("DC energy",edc)
    print("Occupied energies",np.sum(eoccs)/totkp)
    etot = np.sum(eoccs)/totkp + edc  # eigenvalues and double counting
    file_etot.write(str(ite)+"    "+str(etot)+"\n") # write energy in file
    file_error.write(str(ite)+"    "+str(error)+"\n") # write energy in file
    file_etot.flush()
    file_error.flush()
    totcharge = np.sum(charge).real # total charge
    avcharge = totcharge/nat # average charge
    htmp.write_magnetization() # write in a file
    if save: inout.save(mf,mf_file) # save the mean field
    ######
    if not silent:
      print("Times in diagonalization",t2-t1)
      print("Times in new mean field",t3-t2)
      print("\n")
      print("Iteration number =",ite)
      print("Error in SCF =",error)
      print("Fermi energy =",fermi)
      print("Total energy =",etot)
      print("Total charge =",totcharge)
      mx = htmp.extract(name="mx")
      my = htmp.extract(name="my")
      mz = htmp.extract(name="mz")
      print("Mag =",np.sum(mx),np.sum(my),np.sum(mz))
      print("Abs mag =",np.sum(np.abs(mx)),np.sum(np.abs(my)),np.sum(np.abs(mz)))
      print("Average charge =",avcharge)
    old_mf = old_mf*mix + mf*(1.-mix) # mixing
    if error<maxerror or os.path.exists("STOP"): # if converged break
      break
    if ite>=maxite: break # if too many iterations
  file_etot.close() # close file
  file_error.close() # close file
  # output result
  class scftmp(): pass # create an empty class
  scf = scftmp() # empty class
  scf.hamiltonian = htmp.copy() # copy Hamiltonian
  scf.hamiltonian.intra -= fermi*np.identity(ndim) # shift Fermi energy
  scf.total_energy = etot # store total energy
  scf.mf = mf # store mean field matrix
  scf.magnetization = mag # store magnetization
  scf.hamiltonian.write_magnetization() # write magnetization into a file
  return scf # return mean field




def selective_U_matrix(Us,m):
  """Return a mean field matrix, using a different U for each atom"""
  try: Us[0] # try to get an index
  except: return Us*m # return
  if len(Us)!=len(m)//2: raise
  mout = m.copy()
  fac = np.matrix(np.identity(len(Us)*2),dtype=np.complex)
  for i in range(len(Us)):
    fac[2*i,2*i] = Us[i]
    fac[2*i+1,2*i+1] = Us[i]
  return m*fac # return matrix



def selective_U_vector(Us,m):
  """Return a mean field matrix, using a different U for each atom"""
  try:
    Us[0] # try to get an index
    return np.array(Us)*m
  except: return Us*m # return






def magnetic_mean_field(wf,U,collinear=False,totkp=1):
  """Return the mean field matrix"""
  (vdup,vddn,vxc,ndn,nup,xc) = get_udxc(wf,totkp=totkp) # density and XC
  if collinear: mf = ndn + nup
  else: mf = ndn + nup - xc - xc.H
  mf = selective_U_matrix(U,mf.todense()) # new intramatrix mean field
  edc = -np.sum(selective_U_vector(U,vdup*vddn)).real
  edc += np.sum(selective_U_vector(U,vxc*np.conjugate(vxc))).real
  charge = (vdup+vddn).real # total charge
  mag = np.array([vxc.real,vxc.imag,(vdup-vddn)/2.]).transpose().real
  return mf,edc,charge,mag # return mean field and double counting energy











def spinless_mean_field(wf,U,collinear=False,totkp=1):
  """Return the mean field matrix"""
  raise # not finished



def get_udxc(voccs,weight=None,totkp=1):
  """Get up/down densities and corresponding mean field matrices"""
  ndim = voccs.shape[1] # dimension of the matrix
  if weight is not None:
    if len(weight)!=voccs.shape[0]: raise # inconsistent dimensions
  nat = ndim//2 # one half
  pdup = np.array([[2*i,2*i] for i in range(nat)]) # up density
  pddn = pdup + 1 # down density
  pxc = np.array([[2*i,2*i+1] for i in range(nat)]) # exchange
  if weight is None: # no weight
    vdup = correlatorsf90.correlators(voccs,pdup)/totkp
    vddn = correlatorsf90.correlators(voccs,pddn)/totkp
    vxc = correlatorsf90.correlators(voccs,pxc)/totkp
  else: # with weight
    vdup = correlatorsf90.correlators_weighted(voccs,weight,pdup)/totkp
    vddn = correlatorsf90.correlators_weighted(voccs,weight,pddn)/totkp
    vxc = correlatorsf90.correlators_weighted(voccs,pxc)/totkp
  ndn = csc_matrix((vdup,pddn.transpose()),dtype=np.complex,shape=(ndim,ndim))
  nup = csc_matrix((vddn,pdup.transpose()),dtype=np.complex,shape=(ndim,ndim))
  xc = csc_matrix((np.conjugate(vxc),pxc.transpose()),
                           dtype=np.complex,shape=(ndim,ndim))
  return (vdup,vddn,vxc,ndn,nup,xc) # return everything



def hubbardscf_spinless(h,g=1.0,nkp = 100,filling=0.5,mag=None,mix=0.9,
                  maxerror=1e-05,silent=False,mf=None,
                  smearing=None,collinear=False,fermi_shift=0.0):
  """ Solve a selfconsistent Hubbard mean field"""
  mix = 1. - mix
  U = g # redefine
  if h.has_spin: raise # only for spinless systems
  os.system("rm -f STOP") # remove stop file
  from scipy.linalg import eigh
  nat = h.intra.shape[0] # number of atoms
  htmp = h.copy()  # copy hamiltonian
  # generalate the necessary list of correlators
  if mf is None: # generate initial mean field
    mf = np.random.random(h.intra.shape) # random matrix
  else: old_mf = mf # use guess
  # get the pairs for the correlators
  ndim = h.intra.shape[0] # dimension
  totkp = nkp**(h.dimensionality) # total number of kpoints
  file_etot = open("SCF_ENERGY.OUT","w")
  file_error = open("SCF_ERROR.OUT","w")
  ite = 0 # iteration counter
  scf = scfclass(h) # create scf class
  while True: # infinite loop
    ite += 1 # increase counter
    htmp.intra = h.intra + old_mf # add mean field 
    t1 = time.time()
# get eigenvectors
    eigvals,eigvecs,kvectors = htmp.get_eigenvectors(nk=nkp,kpoints=True)
    eigvecs = np.conjugate(eigvecs)
# fermi energy
    t2 = time.time()
    fermi = get_fermi_energy(eigvals,filling,fermi_shift=fermi_shift)
# occupied states
    eoccs,voccs,koccs = get_occupied_states(eigvals,eigvecs,kvectors,fermi)
# spinless mean field
    mf,edc,charge = spinless_mean_field(voccs,U,totkp=totkp)
    t3 = time.time()
    print("Times in diagonalization",t2-t1)
    print("Times in new mean field",t3-t2)
    error = np.max(np.abs(old_mf-mf)) # absolute difference
    # total energy
    etot = np.sum(eoccs)/totkp + edc  # eigenvalues and double counting
    file_etot.write(str(ite)+"    "+str(etot)+"\n") # write energy in file
    file_error.write(str(ite)+"    "+str(error)+"\n") # write energy in file
    totcharge = np.sum(charge).real # total charge
    avcharge = totcharge/nat # average charge
    ######
    if not silent:
      print("\n")
      print("Iteration number =",ite)
      print("Error in SCF =",error)
      print("Fermi energy =",fermi)
      print("Total energy =",etot)
      print("Total charge =",totcharge)
      print("Average charge =",avcharge)
    old_mf = old_mf*mix + mf*(1.-mix) # mixing
    if error<maxerror or os.path.exists("STOP"): # if converged break
      break
    if ite>maxite: break # if too many iterations
  file_etot.close() # close file
  file_error.close() # close file
  # output result
  class scfclass(): pass # create an empty class
  scf = scftmp() # empty class
  scf.hamiltonian = htmp.copy() # copy Hamiltonian
  scf.hamiltonian.intra -= fermi*np.identity(ndim) # shift Fermi energy
  scf.total_energy = etot # store total energy
  scf.mf = mf # store mean field matrix
  scf.magnetization = mag
  scf.hamiltonian.write_magnetization() # write magnetization into a file
