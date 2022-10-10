
# special band structures

from __future__ import print_function
import scipy.linalg as lg
from scipy.sparse import csc_matrix
import scipy.sparse.linalg as slg
from scipy.sparse import issparse
import numpy as np
from . import timing
from . import klist
from . import topology
from . import operators
from .algebra import braket_wAw
from . import algebra

from .limits import densedimension as maxdim
arpack_tol = 1e-8
arpack_maxiter = 10000


def berry_bands(h,klist=None,mode=None,operator=None):
  """Calculate band structure resolved with Berry curvature"""
  ks = [] # list of kpoints
  if mode is not None: # get the mode
    if mode=="sz": operator = operators.get_sz(h)
    else: raise

  fo = open("BANDS.OUT","w")
  for ik in range(len(klist)): # loop over kpoints
    (es,bs) = topology.operator_berry_bands(h,k=klist[ik],operator=operator)
    for (e,b) in zip(es,bs):
      fo.write(str(ik)+"    "+str(e)+"    "+str(b)+"\n")
  fo.close()



def current_bands(h,klist=None):
  """Calcualte the band structure, with the bands"""
  if h.dimensionality != 1: raise # only 1 dimensional
  # go for the rest
  hkgen = h.get_hk_gen() # get generator of the hamiltonian
  if klist is None:  klist = np.linspace(0,1.,100) # generate k points
  fo = open("BANDS.OUT","w") # output file
  from . import current
  fj = current.current_operator(h) # function that generates the operator
  for k in klist: # loop over kpoints
    hk = hkgen(k) # get k-hamiltonian
    jk = fj(k) # get current operator
    evals,evecs = algebra.eigh(hk) # eigenvectors and eigenvalues
    evecs = np.transpose(evecs) # transpose eigenvectors
    for (e,w) in zip(evals,evecs): # do the loop
        waw = braket_wAw(w,jk).real # product
        fo.write(str(k)+"    "+str(e)+"   "+str(waw)+"\n")
  fo.close()









def ket_Aw(A,w):
  return A@w





def get_bands_nd(h,kpath=None,operator=None,num_bands=None,
                    callback=None,central_energy=0.0,nk=400,
                    ewindow=None,
                    output_file="BANDS.OUT",write=True,
                    silent=True):
  """
  Get an n-dimensional bandstructure
  """
  if num_bands is not None:
    if num_bands>(h.intra.shape[0]-1): num_bands=None
  if operator is not None: operator = h.get_operator(operator)
  if num_bands is None: # all the bands
    if operator is not None: 
      def diagf(m): # diagonalization routine
          return algebra.eigh(m) # all eigenvals and eigenfuncs
    else: 
      def diagf(m): # diagonalization routine
          return algebra.eigvalsh(m) # all eigenvals and eigenfuncs
  else: # using arpack
    h = h.copy()
    h.turn_sparse() # sparse Hamiltonian
    def diagf(m):
      eig,eigvec = slg.eigsh(m,k=num_bands,which="LM",sigma=central_energy,
                                  tol=arpack_tol,maxiter=arpack_maxiter)
      if operator is None: return eig
      else: return (eig,eigvec)
  # open file and get generator
  hkgen = h.get_hk_gen() # generator hamiltonian
  kpath = h.geometry.get_kpath(kpath,nk=nk) # generate kpath
  def getek(k):
    """Compute this k-point"""
    out = "" # output string
    hk = hkgen(kpath[k]) # get hamiltonian
    if operator is None:
      es = diagf(hk)
      es = np.sort(es) # sort energies
      for e in es:  # loop over energies
        out += str(k)+"   "+str(e)+"\n" # write in file
      if callback is not None: callback(k,es) # call the function
    else:
      es,ws = diagf(hk)
      ws = ws.transpose() # transpose eigenvectors
      def evaluate(w,k,A): # evaluate the operator
          if type(A)==operators.Operator:
              waw = A.braket(w,k=kpath[k]).real
          elif callable(A):  
            try: waw = A(w,k=kpath[k]) # call the operator
            except: 
              print("Check out the k optional argument in operator")
              waw = A(w) # call the operator
          else: waw = braket_wAw(w,A).real # calculate expectation value
          return waw # return the result
      for (e,w) in zip(es,ws):  # loop over waves
        if callable(ewindow):
            if not ewindow(e): continue # skip iteration
        if isinstance(operator, (list,)): # input is a list
            waws = [evaluate(w,k,A) for A in operator]
        else: waws = [evaluate(w,k,operator)]
        out += str(k)+"   "+str(e)+"  " # write in file
        for waw in waws:  out += str(waw)+"  " # write in file
        out += "\n" # next line
      # callback function in each iteration
      if callback is not None: callback(k,es,ws) # call the function
    return out # return string
  ### Now evaluate the function
  from . import parallel
  if write: f = open(output_file,"w") # open bands file
  if parallel.cores==1: ### single thread ###
    tr = timing.Testimator("BANDSTRUCTURE",silent=silent) # generate object
    esk = "" # empty list
    for k in range(len(kpath)): # loop over kpoints
      tr.remaining(k,len(kpath)) # estimate of the time
      ek = getek(k)
      esk += ek # store
      if write: f.write(ek) # write this kpoint
      if write: f.flush() # flush in file
  else: # parallel run
      esk = parallel.pcall(getek,range(len(kpath))) # compute all
      esk = "".join(esk) # concatenate all
      if write: f.write(esk)
  if write: f.close()
#  print("\nBANDS finished")
  esk = esk.split("\n") # split
  del esk[-1] # remove last one
  esk = np.array([[float(i) for i in ek.split()] for ek in esk]).T
  return esk # return data



def smalleig(m,numw=10,evecs=False):
  """
  Return the smallest eigenvalues using arpack
  """
  tol = arpack_tol
  eig,eigvec = slg.eigsh(m,k=numw,which="LM",sigma=0.0,
                                  tol=tol,maxiter=arpack_maxiter)
  if evecs:  return eig,eigvec.transpose()  # return eigenvectors
  else:  return eig  # return eigenvalues


def lowest_bands(h,nkpoints=100,nbands=10,operator = None,
                   info = False,kpath = None,discard=None):
  """
  Returns the lowest eigenvaleus of the system
  """
  from scipy.sparse import csc_matrix
  if kpath is None: 
    k = klist.default(h.geometry) # default path
  else: k = kpath
  import gc # garbage collector
  fo = open("BANDS.OUT","w")
  if operator is None: # if there is not an operator
    if h.dimensionality==0:  # dot
      eig,eigvec = slg.eigsh(csc_matrix(h.intra),k=nbands,which="LM",sigma=0.0,
                                  tol=arpack_tol,maxiter=arpack_maxiter)
      eigvec = eigvec.transpose() # transpose
      iw = 0
      for i in range(len(eig)):
        if discard is not None: # use the function
          v = eigvec[i] # eigenfunction
          if discard(v): continue
        fo.write(str(iw)+"     "+str(eig[i])+"\n")
        iw += 1 # increase counter
    elif h.dimensionality>0: 
      hkgen = h.get_hk_gen() # get generator
      for ik in k:  # ribbon
        hk = hkgen(ik) # get hamiltonians
        gc.collect() # clean memory
        eig,eigvec = slg.eigsh(hk,k=nbands,which="LM",sigma=0.0)
        del eigvec # clean eigenvectors
        del hk # clean hk
        for e in eig:
          fo.write(str(ik)+"     "+str(e)+"\n")
        if info:  print("Done",ik,end="\r")
    else: raise # ups
  else:  # if there is an operator
    if h.dimensionality==1:
      hkgen = h.get_hk_gen() # get generator
      for ik in k:
        hk = hkgen(ik) # get hamiltonians
        eig,eigvec = slg.eigsh(hk,k=nbands,which="LM",sigma=0.0)
        eigvec = eigvec.transpose() # tranpose the matrix
        if info:  print("Done",ik,end="\r")
        for (e,v) in zip(eig,eigvec): # loop over eigenvectors
          a = braket_wAw(v,operator)
          fo.write(str(ik)+"     "+str(e)+"     "+str(a)+"\n")
  fo.close()




def get_bands_map(h,n=0,**kwargs):
    """Get a 2d map of the band structure"""
    hk = h.get_hk_gen() # Hamiltonian generator
    def f(k):
        e = algebra.eigvalsh(hk(k))[n]
        return e
    from .spectrum import reciprocal_map
    return reciprocal_map(h,f,filename="BANDS_2D.OUT",**kwargs) 

