from .. import parallel
from .. import algebra
from ..klist import kmesh
import numpy as np
import scipy.sparse.linalg as slg

def get_eigenvectors(h,nk=10,kpoints=False,k=None,sparse=False,
        numw=None,energy=0.0):
  from scipy.sparse import csc_matrix as csc
  shape = h.intra.shape
  if numw is not None: sparse = True
  if h.dimensionality==0:
    if not sparse: vv = algebra.eigh(h.intra)
    if sparse: vv = slg.eigsh(csc(h.intra),k=numw,
            which="LM",sigma=energy,tol=1e-5)
    vecs = np.array([v for v in vv[1].transpose()])
    if kpoints: return vv[0],vecs,np.array([[0.,0.,0.] for e in vv[0]])
    else: return vv[0],vecs
  elif h.dimensionality>0:
    f = h.get_hk_gen()
    if k is None:
      kp = kmesh(h.dimensionality,nk=nk) # generate a mesh
    else:  kp = np.array([k]) # kpoint given on input
#    vvs = [lg.eigh(f(k)) for k in kp] # diagonalize k hamiltonian
    nkp = len(kp) # total number of k-points
    if sparse: # sparse Hamiltonians
        fk = lambda k: slg.eigsh(csc(f(k)),k=numw,which="LM",sigma=energy,tol=1e-5)
        vvs = parallel.pcall(fk,kp)
    else: # dense Hamiltonians
      if parallel.cores>1: # in parallel
#        vvs = parallel.multieigh([f(k) for k in kp]) # multidiagonalization
        vvs = parallel.pcall(lambda k: algebra.eigh(f(k)),kp)
      else: vvs = [algebra.eigh(f(k)) for k in kp] # 
    nume = sum([len(v[0]) for v in vvs]) # number of eigenvalues calculated
    eigvecs = np.zeros((nume,h.intra.shape[0]),dtype=np.complex) # eigenvectors
    eigvals = np.zeros(nume) # eigenvalues

    #### New way ####
#    eigvals = np.array([iv[0] for iv in vvs]).reshape(nkp*shape[0],order="F")
#    eigvecs = np.array([iv[1].transpose() for iv in vvs]).reshape((nkp*shape[0],shape[1]),order="F")
#    if kpoints: # return also the kpoints
#      kvectors = [] # empty list
#      for ik in kp: 
#        for i in range(h.intra.shape[0]): kvectors.append(ik) # store
#      return eigvals,eigvecs,kvectors
#    else:
#      return eigvals,eigvecs

    #### Old way, slightly slower but clearer ####
    iv = 0
    kvectors = [] # empty list
    for ik in range(len(kp)): # loop over kpoints
      vv = vvs[ik] # get eigenvalues and eigenvectors
      for (e,v) in zip(vv[0],vv[1].transpose()):
        eigvecs[iv] = v.copy()
        eigvals[iv] = e.copy()
        kvectors.append(kp[ik])
        iv += 1
    if kpoints: # return also the kpoints
#      for iik in range(len(kp)):
#        ik = kp[iik] # store kpoint 
#        for e in vvs[iik][0]: kvectors.append(ik) # store
      return eigvals,eigvecs,kvectors
    else:
      return eigvals,eigvecs
  else:
    raise


