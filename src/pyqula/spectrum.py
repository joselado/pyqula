# library to deal with the spectral properties of the hamiltonian
import numpy as np
import scipy.linalg as lg
import scipy.sparse.linalg as slg
import os
from .operators import operator2list
from . import operators
from . import parallel
from . import kpm
from . import timing
from . import algebra
from . import densitymatrix
from . import filesystem as fs

from .fermisurface import multi_fermi_surface
from .fermisurfacetk.singlefs import fermi_surface

arpack_tol = 1e-5
arpack_maxiter = 10000



def boolean_fermi_surface(h,write=True,output_file="BOOL_FERMI_MAP.OUT",
                    e=0.0,nk=50,nsuper=1,reciprocal=False,
                    delta=None):
    """Calculates the Fermi surface of a 2d system"""
    if h.dimensionality!=2: # continue if two dimensional
        raise ValueError("the boolean Fermi surface is only defined for 2d "
                "Hamiltonians")
    hk_gen = h.get_hk_gen() # gets the function to generate h(k)
    kxs = np.linspace(-nsuper,nsuper,nk)  # generate kx
    kys = np.linspace(-nsuper,nsuper,nk)  # generate ky
    kdos = [] # empty list
    kxout = []
    kyout = []
    if reciprocal: R = h.geometry.get_k2K() # get matrix
    # setup a reasonable value for delta
    if delta is None:
      delta = 8./np.max(np.abs(h.intra))/nk
    rs = [] # real space vectors
    for x in kxs:
      for y in kxs:
        rs.append([x,y,0.])
        kxout.append(x)
        kyout.append(y)
    rs = np.array(rs) # real space vectors
    ks = np.array([R@r for r in rs]) # change of basis
    from .htk.eigenvectors import peigvalsh, hk_matrix_batch
    hks = hk_matrix_batch(hk_gen,ks) # H(k) batch, densified
    es_batch = peigvalsh(hks) # batched numba eigh, shape (nk*nk,n)
    for evals in es_batch: # loop over kpoints
      de = np.abs(evals - e) # difference with respect to fermi
      de = de[de<delta] # energies close to fermi
      if len(de)>0: kdos.append(1.0) # add to the list
      else: kdos.append(0.0) # add to the list
    if write:  # optionally, write in file
      f = open(output_file,"w") 
      for (x,y,d) in zip(kxout,kyout,kdos):
        f.write(str(x)+ "   "+str(y)+"   "+str(d)+"\n")
      f.close() # close the file
    return (kxout,kyout,d) # return result






















from .bandstructure import braket_wAw


def selected_bands2d(h,output_file="BANDS2D_",nindex=[-1,1],
               nk=50,nsuper=1,reciprocal=True,
               operator=None,k0=[0.,0.]):
  """ Calculate a selected bands in a 2d Hamiltonian"""
  if h.dimensionality!=2: # continue if two dimensional
      raise ValueError("selected_bands2d is only for 2d Hamiltonians")
  hk_gen = h.get_hk_gen() # gets the function to generate h(k)
  kxs = np.linspace(-nsuper,nsuper,nk)+k0[0]  # generate kx
  kys = np.linspace(-nsuper,nsuper,nk)+k0[1]  # generate ky
  if reciprocal: R = h.geometry.get_k2K() # get matrix
  else:  R = np.array(np.identity(3)) # get identity
  # setup a reasonable value for delta
  # setup the operator
  operator = operator2list(operator) # convert into a list
  fs.rmglob(output_file+"*") # delete previous files
  fo = [open(output_file+"_"+str(i)+".OUT","w") for i in nindex] # files
  xys = [(x,y) for x in kxs for y in kxs] # all kpoint pairs
  ks = np.array([np.array(R)@np.array([x,y,0.]) for (x,y) in xys]) # change of basis
  if not h.is_sparse: # dense: batch every k-point's H(k) into one numba eigh call
    from .htk.eigenvectors import peigh
    hks = np.array([hk_gen(k) for k in ks],dtype=np.complex128) # H(k) batch
    es_batch,ws_batch = peigh(hks) # batched numba eigh
  for ik,(x,y) in enumerate(xys):
      if not h.is_sparse: evals,waves = es_batch[ik],ws_batch[ik] # eigenvalues
      else: evals,waves = slg.eigsh(hk_gen(ks[ik]),k=max(np.abs(nindex))*2,sigma=0.0,
             tol=arpack_tol,which="LM") # eigenvalues
      waves = waves.transpose() # transpose
      epos,wfpos = [],[] # positive
      eneg,wfneg = [],[] # negative
      for (e,w) in zip(evals,waves): # loop
        if e>0.0: # positive
          epos.append(e)
          wfpos.append(w)
        else: # negative
          eneg.append(e)
          wfneg.append(w)
      # now sort the waves (sort key is the energy only, so degenerate
      # eigenvalues don't fall through to comparing eigenvector arrays)
      wfpos = [yy for (xx,yy) in sorted(zip(epos,wfpos),key=lambda p: p[0])]
      wfneg = [yy for (xx,yy) in sorted(zip(-np.array(eneg),wfneg),key=lambda p: p[0])]
      epos = sorted(epos)
      eneg = -np.array(sorted(-np.array(eneg)))
      for (i,j) in zip(nindex,range(len(nindex))): # loop over desired bands
        fo[j].write(str(x)+"     "+str(y)+"   ")
        if i>0: # positive
          fo[j].write(str(epos[i-1])+"  ")
          for op in operator: # loop over operators
            c = op.braket(wfpos[i-1]).real # expectation value
            fo[j].write(str(c)+"  ") # write in file
          fo[j].write("\n") # write in file
          
        if i<0: # negative
          fo[j].write(str(eneg[abs(i)-1])+"\n")
          for op in operator: # loop over operators
            c = op.braket(wfpos[abs(i)-1]).real # expectation value
            fo[j].write(str(c)+"  ") # write in file
          fo[j].write("\n") # write in file
  [f.close() for f in fo] # close file


get_bands = selected_bands2d




def ev2d(h,nk=50,nsuper=1,reciprocal=False,
               operator=None,k0=[0.,0.],kreverse=False):
  """ Calculate the expectation value of a certain operator"""
  if h.dimensionality!=2: # continue if two dimensional
      raise ValueError("ev2d is only for 2d Hamiltonians")
  hk_gen = h.get_hk_gen() # gets the function to generate h(k)
  kxs = np.linspace(-nsuper,nsuper,nk,endpoint=True)+k0[0]  # generate kx
  kys = np.linspace(-nsuper,nsuper,nk,endpoint=True)+k0[1]  # generate ky
  if kreverse: kxs,kys = -kxs,-kys
  if reciprocal: R = h.geometry.get_k2K() # get matrix
  else:  R = np.array(np.identity(3)) # get identity
  # setup the operator
  operator = operator2list(operator) # convert into a list
  fo = open("EV2D.OUT","w") # open file
  xys = [(x,y) for x in kxs for y in kxs] # all kpoint pairs
  ks = np.array([R@np.array([x,y,0.]) for (x,y) in xys]) # change of basis
  if not h.is_sparse: # dense: batch every k-point's H(k) into one numba eigh call
    from .htk.eigenvectors import peigh
    hks = np.array([hk_gen(k) for k in ks],dtype=np.complex128) # H(k) batch
    es_batch,ws_batch = peigh(hks) # batched numba eigh
  for ik,(x,y) in enumerate(xys):
      print("Doing",x,y)
      if not h.is_sparse: evals,waves = es_batch[ik],ws_batch[ik] # eigenvalues
      else: evals,waves = slg.eigsh(hk_gen(ks[ik]),k=max(nindex)*2,sigma=0.0,
             tol=arpack_tol,which="LM") # eigenvalues
      waves = waves.transpose() # transpose
      eneg,wfneg = [],[] # negative
      for (e,w) in zip(evals,waves): # loop
        if e<0: # negative
          eneg.append(e)
          wfneg.append(w)
      fo.write(str(x)+"     "+str(y)+"   ") # write k-point
      for op in operator: # loop over operators
          c = sum([braket_wAw(w,op) for w in wfneg]).real # expectation value
          fo.write(str(c)+"  ") # write in file
      fo.write("\n") # write in file
  fo.close() # close file





def ev(h,operator=None,nk=30,**kwargs):
  """Calculate the expectation value of a certain number of operators"""
  dm = densitymatrix.full_dm(h,nk=nk,**kwargs)
  if operator is None: # no operator given on input
    operator = [] # empty list
  elif not isinstance(operator,list): # if it is not a list
    operator = [operator] # convert to list
  # densitymatrix.full_dm builds dm[i,j] = sum_occ conj(psi_i) psi_j, the
  # transpose of the usual rho[i,j] = sum_occ psi_i conj(psi_j), so
  # Tr(dm@A) evaluates <A^T> = <A*> rather than <A>. The two agree for
  # every real operator -- the density, sx, sz, a projector -- which is
  # why this went unnoticed, but for a purely imaginary one (sy, and any
  # current/velocity operator i[H,r]) it silently flips the sign.
  # Cross-checked against magnetism.compute_magnetization, an independent
  # implementation that reads the density matrix elements directly.
  dmt = np.transpose(dm) # the standard density matrix
  out = [np.trace(dmt@op) for op in operator]
  out = np.array(out) # return the result
  out = out.reshape(out.shape[0]) # reshape in case there are indexes
  return out # return array



def real_space_vev(h,operator=None,nk=1,nrep=3,name="REAL_SPACE_VEV.OUT",
        **kwargs):
    """Compute the expectation value in real space"""
    if nk>1: # only Gamma point implemented
        raise NotImplementedError("the real-space expectation value is only "
                "implemented at the Gamma point, so nk must be 1")
    dm = densitymatrix.full_dm(h,nk=nk,**kwargs) # Gamma point DM
    if operator is None: operator = np.identity(dm.shape[0],dtype=np.complex128)
    operator = h.get_operator(operator) # convert to operator
    if h.has_eh:
        # with the electron-hole degree of freedom the sum runs over the
        # whole particle-hole-redundant set of negative-energy BdG states,
        # and full2profile below then adds the electron and hole entries of
        # each site: every site came out as exactly 2.0 whatever the
        # density was, destroying all the spatial information. Restricting
        # the operator to the electron sector is the convention get_vev
        # and get_filling_spinful_nambu already use.
        pe = operators.Operator(operators.get_electron(h))
        operator = pe*operator*pe
    # densitymatrix.full_dm builds dm[i,j] = sum_occ conj(psi_i) psi_j, the
    # transpose of the usual rho, so contracting it untransposed evaluates
    # <A*> instead of <A> -- invisible for a real operator, a sign flip for
    # a purely imaginary one such as the valley operator or sy. The same
    # fix spectrum.ev carries a few lines above.
    rho = operator(np.transpose(dm),k=[0.,0.,0.]) # compute the projected DM
    rho = np.diag(rho).real # extract the diagonal
    rho = h.full2profile(rho) # resum if necessary
    h.geometry.write_profile(rho,nrep=nrep,name=name)
    return rho










def total_energy(h,nk=10,nbands=None,use_kpm=False,random=False,
        kp=None,mode="mesh",tol=1e-1,fermi=0.0):
  """Return the total energy"""
  if nbands is None: h = h.get_dense()
  if h.is_sparse and not use_kpm: 
      if nbands is None:
        print("Sparse Hamiltonian but no bands given, taking 20")
        nbands=20
  f = h.get_hk_gen() # get generator
  etot = 0.0 # initialize
  iv = 0
  def enek(k):
    """Compute energy in this kpoint"""
    hk = f(k)  # kdependent hamiltonian
    if use_kpm: # Kernel polynomial method
      return kpm.total_energy(hk,scale=10.,ntries=20,npol=100) # using KPM
    else: # conventional diagonalization
      if nbands is None: vv = algebra.eigvalsh(hk) # diagonalize k hamiltonian
      else: 
          vv,aa = slg.eigsh(hk,k=4*nbands,which="LM",sigma=0.0) 
          vv = -np.sort(-(vv[vv<fermi])) # negative eigenvalues
          vv = vv[0:nbands] # get the negative eigenvlaues closest to EF
      return np.sum(vv[vv<fermi]) # sum energies below fermi energy
  # compute energy using different modes
  if mode in ("mesh","random") and not use_kpm and nbands is None:
    # dense, plain-diagonalization case: batch all k-points into one
    # numba eigh call instead of pcall-ing algebra.eigvalsh per k-point
    from .htk.eigenvectors import peigvalsh, hk_matrix_batch
    if mode=="mesh":
      from .klist import kmesh
      kp = kmesh(h.dimensionality,nk=nk)
    else: # random
      kp = [np.random.random(3) for i in range(nk)] # random points
    mats = hk_matrix_batch(f,kp) # H(k) batch, densified
    es_batch = peigvalsh(mats) # (nk,n) eigenvalues
    etot = np.mean([np.sum(es[es<fermi]) for es in es_batch]) # compute total energy
  elif mode=="mesh":
    from .klist import kmesh
    kp = kmesh(h.dimensionality,nk=nk)
    etot = np.mean(parallel.pcall(enek,kp)) # compute total energy
  elif mode=="random":
    kp = [np.random.random(3) for i in range(nk)] # random points
    etot = np.mean(parallel.pcall(enek,kp)) # compute total eenrgy
  elif mode=="integrate":
    from scipy import integrate
    if h.dimensionality==1: # one dimensional
        etot = integrate.quad(enek,-1.,1.,epsabs=tol,epsrel=tol)[0]
    elif h.dimensionality==2: # two dimensional
        etot = integrate.dblquad(lambda x,y: enek([x,y]),-1.,1.,-1.,1.,
                epsabs=tol,epsrel=tol)[0]
    else:
        raise NotImplementedError("the integrated total energy is only "
                "implemented for 1d and 2d Hamiltonians")
  else:
      raise ValueError("unknown mode; the total energy accepts 'mesh', "
              "'random' and 'integrate'")
  return etot



from .filling import eigenvalues






def reciprocal_map(h,f,nk=40,reciprocal=True,nsuper=1,
        filename="MAP.OUT",
        write=True,verbosity=0,grid=False):
    """ Calculates the reciprocal map of something"""
    if reciprocal: fR = h.geometry.get_k2K_generator()
    else: fR = lambda x: x
    if write: fo = open(filename,"w") # open file
    nt = nk*nk # total number of points
    ik = 0
    ks = [] # list with kpoints
    from . import parallel
    for x in np.linspace(-nsuper,nsuper,nk,endpoint=False):
      for y in np.linspace(-nsuper,nsuper,nk,endpoint=False):
          ks.append([x,y,0.])
    ks = np.array(ks)
    tr = timing.Testimator(filename.replace(".OUT",""),
                maxite=len(ks),silent=verbosity==0)
    def fp(ki): # function to compute the quantity
        if parallel.cores == 1: tr.iterate()
        else: print("Doing",ki)
        k = fR(ki)
        return f(k) # call function
    bs = np.array(parallel.pcall(fp,ks)) # compute all values
    if write:
        for (b,k) in zip(bs,ks): # write everything
            fo.write(str(k[0])+"   "+str(k[1])+"     "+str(b.real))
            fo.write("     "+str(b.imag)+"\n")
            fo.flush()
        fo.close() # close file
    if grid: # if it is a grid
        from .interpolation import points2grid
        kx,ky,bs = points2grid(ks[:,0],ks[:,1],bs,n=int(np.sqrt(len(bs))))
        ks = np.array([kx,ky])
    return ks,bs



def singlet_map(h,nk=40,nsuper=3,mode="abs"):
    """Compute a map with the superconducting singlet pairing"""
    hk = h.get_hk_gen() # get function
    from .superconductivity import extract_pairing
    def f(k): # define function
      m = hk(k) # call Hamiltonian
      (uu,dd,ud) = extract_pairing(m) # extract the pairing
#      return np.abs(ud) # trace
#      return np.sum(np.abs(ud)) # trace
      if mode=="trace": return ud.trace()[0,0] # trace
      elif mode=="det": return np.linalg.det(ud) # trace
      elif mode=="abs": return np.sum(np.abs(ud)) # trace
    reciprocal_map(h,f,nk=nk,nsuper=nsuper,filename="SINGLET_MAP.OUT")



def pairing_map(h,**kwargs):
    """Compute a map with the superconducting singlet pairing"""
    h0 = h.copy()
    h0.remove_nambu()
    h0.setup_nambu_spinor()
    h = h - h0 # get only the pairing
    hk = h.get_hk_gen() # get function
    def f(k): # define function
        m = hk(k) # call Hamiltonian
        es = algebra.eigvalsh(m)
        return np.max(np.abs(es))
    reciprocal_map(h,f,filename="PAIRING_MAP.OUT",**kwargs)



from .filling import set_filling
from .filling import get_fermi_energy



def get_fermi4filling(h,filling,nk=8):
    """Return the fermi energy for a certain filling"""
    if h.has_eh: # this is an approximation, accurate version to be written 
        h0 = h.copy()
        h0.remove_nambu()
        return get_fermi4filling(h0,filling,nk=nk) # workaround
    else:
        es = eigenvalues(h,nk=nk,notime=True)
        return get_fermi_energy(es,filling)

def get_filling_spinful_nambu(h,nk=10,**kwargs):
    """Filling of a spinful Nambu (BdG) Hamiltonian.

    Counting eigenvalues below zero, as the normal-state branch does, is
    meaningless in the Nambu basis: the spectrum is particle-hole
    symmetric, so exactly half of it always sits below zero whatever the
    density is. What the filling actually is, is the weight the negative
    energy BdG states put on the *electron* components,
    n = sum_{E_nk<0} <psi_nk|P_e|psi_nk> / N_k, normalized by the number
    of electron states per unit cell (2 per site, up and down), so that a
    half filled system gives 0.5 like everywhere else in the library."""
    from . import operators
    pe = np.array(operators.get_electron(h).todense()) # electron projector
    (es,ws) = h.get_eigenvectors(nk=nk,**kwargs) # eigenvalues and vectors
    fac = 1./(nk**h.dimensionality) # number of kpoints
    ne = 0.0 # number of electrons per unit cell
    for (e,w) in zip(es,ws): # loop over states
        if e<0.0: ne += np.conjugate(w).dot(pe@w).real # electron weight
    nstates = h.intra.shape[0]//2 # electron states per unit cell
    return ne*fac/nstates # return the filling


def get_filling(h,**kwargs):
    """Get the filling of a Hamiltonian at this energy"""
    if h.check_mode("spinless_nambu"): # spinless Nambu Hamiltonian
        from .sctk import spinless
        return spinless.get_filling(h,**kwargs)
    elif h.check_mode("spinful_nambu"): # spinful Nambu
        return get_filling_spinful_nambu(h,**kwargs)
    else:
        es = eigenvalues(h,**kwargs) # eigenvalues
        es = np.array(es)
        esf = es[es<0.0]
        return len(esf)/len(es) # return filling





def eigenvalues_kmesh(h,nk=20):
    """Get the eigenvalues in a kmesh"""
    if h.dimensionality!=2: # only for 2d
        raise ValueError("eigenvalues_kmesh is only for 2d Hamiltonians")
    ne = h.intra.shape[0] # number of energies per k-point
    hkgen = h.get_hk_gen() # get the generator
    kx = np.linspace(0.,1.,nk,endpoint=False)
    ky = np.linspace(0.,1.,nk,endpoint=False)
    from .htk.eigenvectors import peigvalsh, hk_matrix_batch
    mats = hk_matrix_batch(hkgen,[[ik,jk] for ik in kx for jk in ky])
    # H(k) batch, ik outer, jk inner
    es_batch = peigvalsh(mats) # batched numba eigh, shape (nk*nk,ne)
    es = es_batch.reshape(nk,nk,ne) # reshape to match original layout
    return es # return all the energies




def lowest_energies(h,n=4,k=None,**kwargs):
    """Return the lowest energy states in a k-point"""
    if k is None:
        raise ValueError("lowest_energies needs the k-point to evaluate, pass "
                "it as k")
    es,ws = h.get_eigenvectors(kpoints=False,k=k,numw=2*n,**kwargs)
    es = [y for (x,y) in sorted(zip(np.abs(es),es))][0:n]
    es = np.sort(es)
    return es



def get_bandwidth(self,**kwargs):
    """Return the bandwidth of the Hamiltonian"""
    from .gap import optimize_energy 
    self = self.get_dense() # dense matrix
    emin = optimize_energy(self,mode="bottom",**kwargs)
    emax = optimize_energy(self,mode="top",**kwargs)
    return (emin,emax)



