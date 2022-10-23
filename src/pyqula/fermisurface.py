# library to deal with the spectral properties of the hamiltonian
import numpy as np
import scipy.linalg as lg
import scipy.sparse.linalg as slg
import os
from .operators import operator2list
from . import parallel
from . import kpm
from . import timing
from . import algebra
from . import filesystem as fs

arpack_tol = 1e-5
arpack_maxiter = 10000

def fermi_surface_generator(h,
                    energies=[0.0],nk=50,nsuper=1,reciprocal=True,
                    delta=1e-2,refine_delta=1.0,operator=None,
                    full_bz=False, # special flag for unfolding
                    numw=20,info=True):
  """Calculates the Fermi surface of a 2d system"""
  if h.is_sparse: mode = "sparse"
  else: mode = "full"
  energies = np.array(energies) # convert to array
  if h.dimensionality!=2: raise  # continue if two dimensional
  hk_gen = h.get_hk_gen() # gets the function to generate h(k)
  if full_bz:
      kxs = np.linspace(0.,nsuper,nk,endpoint=True)  # generate kx
      kys = np.linspace(0.,nsuper,nk,endpoint=True)  # generate ky
  else:
      kxs = np.linspace(-nsuper,nsuper,nk,endpoint=True)  # generate kx
      kys = np.linspace(-nsuper,nsuper,nk,endpoint=True)  # generate ky
  iden = np.identity(h.intra.shape[0],dtype=np.complex)
  kxout = []
  kyout = []
  if reciprocal: fR = h.geometry.get_k2K_generator() # get matrix
  else:  fR = lambda x: x # get identity
  # setup a reasonable value for delta
  #### function to calculate the weight ###
  operator = h.get_operator(operator) # overwrite operator
  def get_weight(hk,k=None):
      if operator is None:
          if mode=='full': es = algebra.eigvalsh(hk) # get eigenvalues
          elif mode=='sparse': es = algebra.smalleig(hk,numw=numw)
          ws = [np.sum(delta/((e-es)**2+delta**2)) for e in energies] # weights
          return np.array(ws) # return weights
      else:
          tmp,ds = h.get_dos(ks=[k],operator=operator,
                        energies=energies,delta=delta)
          return ds # return weight
##############################################
  ts = timing.Testimator()
  # setup the operator
  rs = [] # empty list
  for x in kxs:
    for y in kys:
      rs.append([x,y,0.]) # store
  def getf(r): # function to compute FS
      k = fR(r) # get the reciprocal space vector
      hk = hk_gen(k) # get hamiltonian
      return get_weight(hk,k=k) # get the array with the weights
  rs = np.array(rs) # transform into array
  from . import parallel
  kxout = rs[:,0] # x coordinate
  kyout = rs[:,1] # y coordinate
  if parallel.cores==1: # serial execution
      kdos = [] # empty list
      for r in rs: # loop
        if info: print("Doing",r)
        kdos.append(getf(r)) # add to the list
  else: # parallel execution
      kdos = parallel.pcall(getf,rs) # compute all
  kdos = np.array(kdos) # transform into an array
  return energies,rs,kdos


def multi_fermi_surface(h,nk=None,delta=1e-2,
        output_folder="MULTIFERMISURFACE",**kwargs):
    """Compute several fermi surfaces"""
    if nk is None: nk = int(10./delta)
    energies,rs,kdos = fermi_surface_generator(h,nk=nk,delta=delta,**kwargs)
    fs.rmdir(output_folder) # remove folder
    fs.mkdir(output_folder) # create folder
    fo = open(output_folder+"/"+output_folder+".TXT","w")
    for i in range(len(energies)): # loop over energies
        filename = output_folder+"_"+str(energies[i])+"_.OUT" # name
        name = output_folder+"/"+filename
        np.savetxt(name,np.array([rs[:,0],rs[:,1],kdos[:,i]]).T)
        fo.write(filename+"\n")
    name = output_folder+"/DOS.OUT"
    name = "DOS.OUT"
    np.savetxt(name,np.array([energies,np.sum(kdos,axis=0)]).T)
    fo.close()


