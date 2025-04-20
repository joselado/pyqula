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

from .fermisurface import multi_fermi_surface

arpack_tol = 1e-5
arpack_maxiter = 10000

def fermi_surface(h,write=True,output_file="FERMI_MAP.OUT",
                    e=0.0,nk=50,nsuper=1,reciprocal=True,
                    k0 = np.array([0.,0.]),
                    delta=None,refine_delta=1.0,operator=None,
                    mode='eigen',num_waves=2,info=False):
    """Calculates the Fermi surface of a 2d system"""
    operator = h.get_operator(operator) # get the operator
    if operator is not None: # operator given
        if not operator.linear:
            if mode=="full": mode = "eigen"
    else: # no operator given
        if mode=="full":
            operator = np.matrix(np.identity(h.intra.shape[0]))
    if h.dimensionality!=2: raise  # continue if two dimensional
    hk_gen = h.get_hk_gen() # gets the function to generate h(k)
    kxs = np.linspace(-nsuper,nsuper,nk)  # generate kx
    kys = np.linspace(-nsuper,nsuper,nk)  # generate ky
    iden = np.identity(h.intra.shape[0],dtype=np.complex128)
    kxout = []
    kyout = []
    if reciprocal: R = h.geometry.get_k2K_generator() # get function
    else:  R = lambda x: x
    # setup a reasonable value for delta
    if delta is None:  delta = 3./refine_delta*2./nk
    #### function to calculate the weight ###
    if mode=='full': # use full inversion
      def get_weight(hk,k=None):
        gf = algebra.inv((e+1j*delta)*iden - hk) # get green function
        gf = gf - algebra.inv((e-1j*delta)*iden - hk) # get green function
        if callable(operator): # callable operator
           tdos = -(operator(gf,k=k)).imag # get imaginary part
        else: tdos = -(operator*gf).imag # get imaginary part
        return np.trace(tdos).real # return trace
    elif mode=='eigen': # use full diagonalization
      def get_weight(hk,k=None):
        if operator is None:
            es = algebra.eigvalsh(hk)
            return np.sum(delta/((e-es)**2+delta**2)) # return weight
        else: # using an operator
            tmp,ds = h.get_dos(ks=[k],operator=operator,
                        energies=[e],delta=delta)
            return ds[0] # return weight
    elif mode=='lowest': # use sparse diagonalization
      def get_weight(hk,**kwargs):
        es,waves = slg.eigsh(hk,k=num_waves,sigma=e,tol=arpack_tol,which="LM",
                              maxiter = arpack_maxiter)
        return np.sum(delta/((e-es)**2+delta**2)) # return weight
    else: raise
  
  ##############################################
  
  
    # setup the operator
    rs = [] # empty list
    for x in kxs:
      for y in kxs:
        rs.append([x,y,0.]) # store
    k0 = np.array([k0[0],k0[1],0.])
    def getf(r): # function to compute FS
        k = R(r) + k0
        hk = hk_gen(k) # get hamiltonian
        return get_weight(hk,k=k)
    rs = np.array(rs) # transform into array
    from . import parallel
    kxout = rs[:,0] # x coordinate
    kyout = rs[:,1] # y coordinate
    if parallel.cores==1: # serial execution
        if info: ts = timing.Testimator(maxite=len(kxs)*len(kys),title="FS")
        kdos = [] # empty list
        for r in rs: # loop
          if info: ts.iterate()
          kdos.append(getf(r)) # add to the list
    else: # parallel execution
        kdos = parallel.pcall(getf,rs) # compute all
    if write:  # optionally, write in file
      f = open(output_file,"w") 
      for (x,y,d) in zip(kxout,kyout,kdos):
        f.write(str(x)+ "   "+str(y)+"   "+str(d)+"\n")
      f.close() # close the file
    return (kxout,kyout,np.array(kdos)) # return result




def boolean_fermi_surface(h,write=True,output_file="BOOL_FERMI_MAP.OUT",
                    e=0.0,nk=50,nsuper=1,reciprocal=False,
                    delta=None):
    """Calculates the Fermi surface of a 2d system"""
    if h.dimensionality!=2: raise  # continue if two dimensional
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
    for x in kxs:
      for y in kxs:
        r = np.matrix([x,y,0.]).T # real space vectors
        k = np.array((R*r).T)[0] # change of basis
        hk = hk_gen(k) # get hamiltonian
        evals = lg.eigvalsh(hk) # diagonalize
        de = np.abs(evals - e) # difference with respect to fermi
        de = de[de<delta] # energies close to fermi
        if len(de)>0: kdos.append(1.0) # add to the list
        else: kdos.append(0.0) # add to the list
        kxout.append(x)
        kyout.append(y)
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
  if h.dimensionality!=2: raise  # continue if two dimensional
  hk_gen = h.get_hk_gen() # gets the function to generate h(k)
  kxs = np.linspace(-nsuper,nsuper,nk)+k0[0]  # generate kx
  kys = np.linspace(-nsuper,nsuper,nk)+k0[1]  # generate ky
  kdos = [] # empty list
  kxout = []
  kyout = []
  if reciprocal: R = h.geometry.get_k2K() # get matrix
  else:  R = np.array(np.identity(3)) # get identity
  # setup a reasonable value for delta
  # setup the operator
  operator = operator2list(operator) # convert into a list
  os.system("rm -f "+output_file+"*") # delete previous files
  fo = [open(output_file+"_"+str(i)+".OUT","w") for i in nindex] # files        
  for x in kxs:
    for y in kxs:
#      print("Doing",x,y)
      r = np.array([x,y,0.]) # real space vectors
      k = np.array(R)@r # change of basis
#      print(k)
      hk = hk_gen(k) # get hamiltonian
      if not h.is_sparse: evals,waves = lg.eigh(hk) # eigenvalues
      else: evals,waves = slg.eigsh(hk,k=max(np.abs(nindex))*2,sigma=0.0,
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
      # now sort the waves
      wfpos = [yy for (xx,yy) in sorted(zip(epos,wfpos))] 
      wfneg = [yy for (xx,yy) in sorted(zip(-np.array(eneg),wfneg))] 
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
  if h.dimensionality!=2: raise  # continue if two dimensional
  hk_gen = h.get_hk_gen() # gets the function to generate h(k)
  kxs = np.linspace(-nsuper,nsuper,nk,endpoint=True)+k0[0]  # generate kx
  kys = np.linspace(-nsuper,nsuper,nk,endpoint=True)+k0[1]  # generate ky
  if kreverse: kxs,kys = -kxs,-kys
  kdos = [] # empty list
  kxout = []
  kyout = []
  if reciprocal: R = h.geometry.get_k2K() # get matrix
  else:  R = np.matrix(np.identity(3)) # get identity
  # setup the operator
  operator = operator2list(operator) # convert into a list
  fo = open("EV2D.OUT","w") # open file
  for x in kxs:
    for y in kxs:
      print("Doing",x,y)
      r = np.matrix([x,y,0.]).T # real space vectors
      k = np.array((R*r).T)[0] # change of basis
      hk = hk_gen(k) # get hamiltonian
      if not h.is_sparse: evals,waves = lg.eigh(hk) # eigenvalues
      else: evals,waves = slg.eigsh(hk,k=max(nindex)*2,sigma=0.0,
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
  out = [np.trace(dm@op) for op in operator] 
  out = np.array(out) # return the result
  out = out.reshape(out.shape[0]) # reshape in case there are indexes
  return out # return array



def real_space_vev(h,operator=None,nk=1,nrep=3,name="REAL_SPACE_VEV.OUT",
        **kwargs):
    """Compute the expectation value in real space"""
    if nk>1: raise # only Gamma point implemented
    dm = densitymatrix.full_dm(h,nk=nk,**kwargs) # Gamma point DM
    operator = operators.Operator(operator) # convert to operator
    rho = operator(dm,k=[0.,0.,0.]) # compute the projected DM
    rho = np.diag(rho).real # extract the diagonal
    rho = h.full2profile(rho) # resum if necessary
    h.geometry.write_profile(rho,nrep=5,name=name)
    return rho










def total_energy(h,nk=10,nbands=None,use_kpm=False,random=False,
        kp=None,mode="mesh",tol=1e-1,fermi=0.0):
  """Return the total energy"""
  if nbands is None: h.turn_dense()
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
  if mode=="mesh":
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
    else: raise
  else: raise
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

def get_filling(h,**kwargs):
    """Get the filling of a Hamiltonian at this energy"""
    if h.check_mode("spinless_nambu"): # spinless Nambu Hamiltonian
        from .sctk import spinless
        return spinless.get_filling(h,**kwargs)
    elif h.check_mode("spinful_nambu"): raise # spinful Nambu
    else:
        es = eigenvalues(h,**kwargs) # eigenvalues
        es = np.array(es)
        esf = es[es<0.0]
        return len(esf)/len(es) # return filling





def eigenvalues_kmesh(h,nk=20):
    """Get the eigenvalues in a kmesh"""
    if h.dimensionality!=2: raise # only for 2d
    ne = h.intra.shape[0] # number of energies per k-point
    es = np.zeros((nk,nk,ne)) # array for the energies 
    hkgen = h.get_hk_gen() # get the generator
    kx = np.linspace(0.,1.,nk,endpoint=False)
    ky = np.linspace(0.,1.,nk,endpoint=False)
    for i in range(nk):
      ik = kx[i] # kx point   
      for j in range(nk):
        jk = ky[j] # ky point
        hk = hkgen([ik,jk]) # get the matrix
        ei = algebra.eigvalsh(hk) # get the energies
        es[i,j,:] = ei # store energies
    return es # return all the energies




def lowest_energies(h,n=4,k=None,**kwargs):
    """Return the lowest energy states in a k-point"""
    if k is None: raise
    es,ws = h.get_eigenvectors(kpoints=False,k=k,numw=2*n,**kwargs)
    es = [y for (x,y) in sorted(zip(np.abs(es),es))][0:n]
    es = np.sort(es)
    return es



def get_bandwidth(self,**kwargs):
    """Return the bandwidth of the Hamiltonian"""
    from .gap import optimize_energy 
    self.turn_dense() # dense matrix
    emin = optimize_energy(self,mode="bottom",**kwargs)
    emax = optimize_energy(self,mode="top",**kwargs)
    return (emin,emax)



