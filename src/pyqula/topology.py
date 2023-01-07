# library to calculate topological properties
import numpy as np
from scipy.sparse import bmat, csc_matrix
import scipy.linalg as lg
import scipy.sparse.linalg as slg
from . import multicell
from . import klist
from . import operators
from . import inout
from . import timing
from . import algebra
from . import parallel

arpack_tol = algebra.arpack_tol
arpack_maxiter = algebra.arpack_maxiter


def write_berry(h,kpath=None,dk=0.01,window=None,max_waves=None,nk=600,
      mode="Wilson",delta=0.001,reciprocal=False,operator=None,
      silent = True):
    """Calculate and write in file the Berry curvature"""
    operator = get_operator(h,operator)
    kpath = klist.get_kpath(h.geometry,kpath=kpath,nk=nk) # take default kpath
    tr = timing.Testimator("BERRY CURVATURE",silent=silent)
    ik = 0
    if operator is not None: mode="Green" # Green function mode
    def getb(k):
      if reciprocal:  k = h.geometry.get_k2K_generator()(k) # convert
      if mode=="Wilson":
        b = berry_curvature(h,k,dk=dk,window=window,max_waves=max_waves)
      elif mode=="Green":
        f = h.get_gk_gen(delta=delta) # get generator
        b = berry_green(f,k=k,operator=operator) 
      else: raise
      return str(k[0])+"   "+str(k[1])+"   "+str(b)+"\n"
    fo = open("BERRY_CURVATURE.OUT","w") # open file
    if parallel.cores==1: # serial execution
      for k in kpath:
        tr.remaining(ik,len(kpath))
        ik += 1
        fo.write(getb(k)) # write result
        fo.flush()
    else: # parallel execution
        out = parallel.pcall(getb,kpath)
        for o in out: fo.write(o) # write
    fo.close() # close file
    m = np.genfromtxt("BERRY_CURVATURE.OUT").transpose()
    return np.array(range(len(m[0]))),m[2]










def berry_phase(h,nk=20,kpath=None,write=True):
    """ Calculates the Berry phase of a Hamiltonian"""
    if h.dimensionality==0: raise
    elif h.dimensionality == 1:
      ks = np.linspace(0.,1.,nk,endpoint=False) # list of kpoints
    elif h.dimensionality > 1: # you must provide a kpath
        if kpath is None: 
            print("You must provide a k-path")
            raise # error
        ks = kpath # continue
        nk = len(kpath) # redefine
    else: raise # otherwise
    hkgen = h.get_hk_gen() # get Hamiltonian generator
    wf0 = occupied_states(hkgen,ks[0]) # get occupied states, first k-point
    wfold = wf0.copy() # copy
    m = np.matrix(np.identity(len(wf0))) # initialize as the identity matrix
    for ik in range(1,len(ks)): # loop over k-points, except first one
      wf = occupied_states(hkgen,ks[ik])  # get waves
      m = m@uij(wfold,wf)   # get the uij   and multiply
      wfold = wf.copy() # this is the new old
    m = m@uij(wfold,wf0)   # last one
    d = lg.det(m) # calculate determinant
    phi = np.arctan2(d.imag,d.real)
    if write: open("BERRY_PHASE.OUT","w").write(str(phi/np.pi)+"\n")
    return phi # return Berry phase








def berry_curvature(h,k,dk=0.01,window=None,max_waves=None):
  """ Calculates the Berry curvature of a 2d hamiltonian"""
  if h.dimensionality != 2: raise # only for 2d
  k = np.array([k[0],k[1]]) 
  dx = np.array([dk,0.])
  dy = np.array([0.,dk])
# get the function that returns the occ states
  occf = occ_states_generator(h,k,window=window,max_waves=max_waves)  
  # get the waves
#  print("Doing k-point",k)
  wf1 = occf(k-dx-dy) 
  wf2 = occf(k+dx-dy) 
  wf3 = occf(k+dx+dy) 
  wf4 = occf(k-dx+dy) 
  dims = [len(wf1),len(wf2),len(wf3),len(wf4)] # number of vectors
  if max(dims)!=min(dims): # check that the dimensions are fine 
#    print("WARNING, skipping this k-point",k)
    return 0.0 # if different number of vectors
  # get the uij  
  m = uij(wf1,wf2)@uij(wf2,wf3)@uij(wf3,wf4)@uij(wf4,wf1)
  d = lg.det(m) # calculate determinant
  phi = np.arctan2(d.imag,d.real)/(4.*dk*dk)
  return phi


from .topologytk.occstates import occ_states_generator
from .topologytk.occstates import occupied_states
from .topologytk.occstates import occ_states2d




def uij(wf1,wf2):
  """ Calcultes the matrix product of two sets of input wavefunctions"""
  out =  np.matrix(np.conjugate(wf1))@(np.matrix(wf2).T)  # faster way
  return out


def uij_slow(wf1,wf2):
  m = np.matrix(np.zeros((len(wf1),len(wf2)),dtype=np.complex))
  for i in range(len(wf1)):
    for j in range(len(wf2)):
      m[i,j] = np.conjugate(wf1[i]).dot(wf2[j])
  return m


def precise_chern(h,dk=0.01, mode="Wilson",delta=0.0001,operator=None):
  """ Calculates the chern number of a 2d system """
  from scipy import integrate
  err = {"epsabs" : 1.0, "epsrel": 1.0,"limit" : 10}
#  err = [err,err]
  def f(x,y): # function to integrate
    if mode=="Wilson":
      return berry_curvature(h,np.array([x,y]),dk=dk)
    if mode=="Green":
       f2 = h.get_gk_gen(delta=delta) # get generator
       return berry_green(f2,k=[x,y,0.],operator=operator) 
  c = integrate.dblquad(f,0.,1.,lambda x : 0., lambda x: 1.,epsabs=0.01,
                          epsrel=0.01)
  chern = c[0]/(2.*np.pi)
  open("CHERN.OUT","w").write(str(chern)+"\n")
  return chern


def hall_conductivity(h,dk=-1,n=1000):
  c = 0.0 
  nk = int(np.sqrt(n)) # estimate
  if dk<0: dk = 1./float(2*nk) # automatic dk
  for i in range(n):
    k = np.random.random(2) # random kpoint
    c += berry_curvature(h,k,dk=dk)
  c = c/(2*np.pi*n) # normalize
  return c




def mesh_chern(h,dk=-1,nk=10,delta=0.0001,mode="Wilson",
        operator=None,kmesh=None):
  """ Calculates the chern number of a 2d system """
  c = 0.0
  ks = [] # array for kpoints
  bs = [] # array for berrys
  if dk<0: dk = 1./float(2*nk) # automatic dk
  if kmesh is not None: # infer the dk of the mesh
      dk = klist.infer_kmesh_dk(kmesh,d=2)
  if operator is not None and mode=="Wilson":
    print("Switching to Green mode in topology")
    mode="Green"
  # create the function
  def fberry(k): # function to integrate
    if mode=="Wilson":
      return berry_curvature(h,k,dk=dk)
    if mode=="Green":
       f2 = h.get_gk_gen(delta=delta) # get generator
       return berry_green(f2,k=[k[0],k[1],0.],operator=operator) 
  ##################
  if kmesh is None: # no kmesh provided
      ks = klist.kmesh(h.dimensionality,nk=nk) # get the mesh
  else: ks = kmesh # use the provided kmesh
  ik = 0
  bs = parallel.pcall(fberry,ks) # compute all the Berry curvatures
  # write in file
  fo = open("BERRY_CURVATURE.OUT","w") # open file
  for (k,b) in zip(ks,bs):
    fo.write(str(k[0])+"   ")
    fo.write(str(k[1])+"   ")
    fo.write(str(b)+"\n")
  fo.close() # close file
  ################
  c = np.sum(bs) # sum berry curvatures
  if kmesh is None: # no kmesh provided
      c = c/(2.*np.pi*nk*nk) # normalize
  else: # kmesh is given
      den = klist.infer_kmesh_density(kmesh,d=2) # infer the volume
      c = den*c/(2.*np.pi) # normalize
  open("CHERN.OUT","w").write(str(c)+"\n")
  return c



def get_berry_curvature(h,dk=None,nk=100,reciprocal=True,nsuper=1,window=None,
               max_waves=None,mode="Wilson",delta=0.001,operator=None,
               write=True,verbose=0):
    """ Return the Berry curvature in 2D reciprocal space """
    if operator is not None: mode="Green" # Green function mode
    c = 0.0
    ks = [] # array for kpoints
    if dk is None: dk = 1./float(2*nk) # automatic dk
    if reciprocal: R = np.array(h.geometry.get_k2K())
    else: R = np.array(np.identity(3))
    nt = nk*nk # total number of points
    ik = 0
    ks = [] # list with kpoints
    from . import parallel
    for x in np.linspace(-nsuper,nsuper,nk,endpoint=False):
      for y in np.linspace(-nsuper,nsuper,nk,endpoint=False):
          ks.append([x,y,0.])
    ks = np.array(ks) # convert to array
    if verbose>0: tr = timing.Testimator("BERRY CURVATURE",maxite=len(ks))
    def fp(ki): # function to compute the Berry curvature
        if parallel.cores == 1: 
            if verbose>0: tr.iterate()
        else: 
            if verbose>0:  print("Doing",ki)
        k = R@ki # change of basis
        if mode=="Wilson":
           b = berry_curvature(h,k,dk=dk,window=window,max_waves=max_waves)
        elif mode=="Green":
           f = h.get_gk_gen(delta=delta) # get generator
           b = berry_green(f,k=k,operator=operator) 
        else: raise
        return b
    bs = parallel.pcall(fp,ks) # compute all the Berry curvatures
    if write: # write result in a file
        fo = open("BERRY_MAP.OUT","w") # open file
        for (b,k) in zip(bs,ks): # write everything
            fo.write(str(k[0])+"   "+str(k[1])+"     "+str(b)+"\n")
            fo.flush()
        fo.close() # close file
    return [ks[:,0],ks[:,1],np.array(bs)] # return result


berry_map = get_berry_curvature # alias


def smooth_gauge(w1,w2):
  """Perform a gauge rotation so that the second set of waves are smooth
  with respect to the first one"""
  m = uij(w1,w2) # matrix of wavefunctions
  U, s, V = np.linalg.svd(m, full_matrices=True) # sing val decomp
  R = (U@V).H # rotation matrix
  wnew = w2.copy()*0j # initialize
  wold = w2.copy() # old waves
  for ii in range(R.shape[0]):
    for jj in range(R.shape[0]):
      wnew[ii] += R[jj,ii]*wold[jj]
  return wnew




def z2_vanderbilt(h,nk=30,nt=100,nocc=None,full=False):
  """ Calculate Z2 invariant according to Vanderbilt algorithm"""
  out = [] # output list
  path = np.linspace(0.,1.,nk) # set of kpoints
  fo = open("WANNIER_CENTERS.OUT","w")
  if full:  ts = np.linspace(0.,1.0,nt,endpoint=False)
  else:  ts = np.linspace(0.,0.5,nt,endpoint=False)
  wfall = [[occ_states2d(h,np.array([k,t,0.,])) for k in path] for t in ts] 
  # select a continuos gauge for the first wave
  for it in range(len(ts)-1): # loop over ts
    wfall[it+1][0] = smooth_gauge(wfall[it][0],wfall[it+1][0]) 
  for it in range(len(ts)): # loop over t points
    row = [] # empty list for this row
    t = ts[it] # select the t point
    wfs = wfall[it] # get set of waves 
    for i in range(len(wfs)-1):
      wfs[i+1] = smooth_gauge(wfs[i],wfs[i+1]) # transform into a smooth gauge
#      m = uij(wfs[i],wfs[i+1]) # matrix of wavefunctions
    m = uij(wfs[0],wfs[len(wfs)-1]) # matrix of wavefunctions
    evals = lg.eigvals(m) # eigenvalues of the rotation 
    x = np.angle(evals) # phase of the eigenvalues
    fo.write(str(t)+"    ") # write pumping variable
    row.append(t) # store
    for ix in x: # loop over phases
      fo.write(str(ix)+"  ")
      row.append(ix) # store
    fo.write("\n")
    out.append(row) # store
  fo.close()
  return np.array(out).transpose() # transpose the map


def z2_invariant(h,nk=20,nt=20,nocc=None):
  """Compute Z2 invariant with pumping of Wannier centers"""
  return wannier_winding(h,nk=nk,nt=nt,nocc=nocc,full=False) 




def chern(h,**kwargs):
  """Compute Chern invariant"""
  return mesh_chern(h,**kwargs) # workaround
  # the wannier winding does not work
  c = wannier_winding(h,full=True,**kwargs) 
  open("CHERN.OUT","w").write(str(c))
  return c




def wannier_winding(h,nk=40,nt=40,nocc=None,full=True):
  m = z2_vanderbilt(h,nk=nk,nt=nt,nocc=nocc,full=full)
  x = m[0]
  # find the position of the maximum gap at every t
  fermis = x*0. # maximum gap
  for it in range(len(x)): # loop over times
    imax,jmax = None,None
    dmax = -1 # initialize
    gapangle = None # initialize
    maxgap = -1.0 # maximum gap
    gapangle = ((m[1][it]+0.5)%1)*np.pi # initialize
    for i in range(1,len(m)):
      for j in range(i+1,len(m)):
        for ipi in [0.,1.]:
          ip = np.exp(1j*m[i][it]) # center of wave i
          jp = np.exp(1j*m[j][it]) # center of wave j
          angle = np.angle(ip+jp)+np.pi*ipi # get the angle
          dp = np.exp(1j*angle) # now obtain this middle point gap
          mindis = 4.0 # calculate minimum distance
          for k in range(1,len(m)): # loop over centers
            kp = np.exp(1j*m[k][it]) # center of wave k
            dis = np.abs(dp-kp) # distance between the two points 
            if dis<mindis: mindis = dis+0. # update minimum distance
          if mindis>maxgap: # if found a bigger gap
            maxgap = mindis+0. # maximum distance
            gapangle = np.angle(dp) # update of found bigger gap
    fermis[it] = gapangle
  # now check the number of cuts of each wannier center
  def angleg(a,b,c):
    """Function to say if a jump has been made or not"""
    d = np.sin(a-b) + np.sin(b-c) + np.sin(c-a)
    return -d

  if full: # for the Chern number
    raise
    # this part is wrong
    #####################
    cuts = 0 # start with 0
#    print(fermis)
    for i in range(1,len(m)): # loop over waves 
      cwf = m[i] # center of the wave
      for it in range(len(x)-1): # loop over times
        s1 = np.sign(fermis[it]-cwf[it])
        s2 = np.sign(fermis[it+1]-cwf[it])
#        print(s1,s2)
        cuts += (s1-s2)/2. 
    return cuts

  else: # for the Z2 invariant
    parity = 1 # start with
    for i in range(1,len(m)): # loop over waves 
      cwf = m[i] # center of the wave
      for it in range(len(x)-1): # loop over times
        s = np.sign(angleg(fermis[it],fermis[it+1],cwf[it])) 
        if s<0.:  parity *= -1 # add a minus sign
    return parity










def operator_berry(hin,k=[0.,0.],operator=None,delta=0.00001,ewindow=None):
  """Calculates the Berry curvature using an arbitrary operator"""
  h = multicell.turn_multicell(hin) # turn to multicell form
  dhdx = multicell.derivative(h,k,order=[1,0]) # derivative
  dhdy = multicell.derivative(h,k,order=[0,1]) # derivative
  hkgen = h.get_hk_gen() # get generator
  hk = hkgen(k) # get hamiltonian
  (es,ws) = algebra.eigh(hkgen(k)) # initial waves
  ws = np.conjugate(np.transpose(ws)) # transpose the waves
  n = len(es) # number of energies
  from .berry_curvaturef90 import berry_curvature as bc90
  if operator is None: operator = np.identity(dhdx.shape[0],dtype=np.complex)
  b = bc90(dhdx,dhdy,ws,es,operator,delta) # berry curvature
  return b*np.pi*np.pi*8 # normalize so the sum is 2pi Chern



def operator_berry_bands(hin,k=[0.,0.],operator=None,delta=0.00001):
  """Calculates the Berry curvature using an arbitrary operator"""
  h = multicell.turn_multicell(hin) # turn to multicell form
  dhdx = multicell.derivative(h,k,order=[1,0]) # derivative
  dhdy = multicell.derivative(h,k,order=[0,1]) # derivative
  hkgen = h.get_hk_gen() # get generator
  hk = hkgen(k) # get hamiltonian
  (es,ws) = algebra.eigh(hkgen(k)) # initial waves
  ws = np.conjugate(np.transpose(ws)) # transpose the waves
  from .berry_curvaturef90 import berry_curvature_bands as bcb90
  if operator is None: operator = np.identity(dhdx.shape[0],dtype=np.complex)
  bs = bcb90(dhdx,dhdy,ws,es,operator,delta) # berry curvatures
  return (es,bs*np.pi*np.pi*8) # normalize so the sum is 2pi Chern







def spin_chern(h,nk=40,delta=0.00001,k0=[0.,0.],expandk=1.0):
  """Calculate the spin Chern number"""
  kxs = np.linspace(-.5,.5,nk,endpoint=False)*expandk + k0[0]
  kys = np.linspace(-.5,.5,nk,endpoint=False)*expandk + k0[1]
  kk = [] # list of kpoints
  for i in kxs:
    for j in kys:
      kk.append(np.array([i,j])) # store vector
  sz = operators.get_sz(h) # get sz operator
  bs = [operator_berry(h,k=ki,operator=sz,delta=delta) for ki in kk] # get all berries
  fo = open("BERRY_CURVATURE_SZ.OUT","w") # open file
  for (k,b) in zip(kk,bs):
    fo.write(str(k[0])+"   ")
    fo.write(str(k[1])+"   ")
    fo.write(str(b)+"\n")
  fo.close() # close file
  bs = np.array(bs)/(2.*np.pi) # normalize by 2 pi
  return sum(bs)/len(kk)


def write_spin_berry(h,kpath,delta=0.00001,operator=None):
  """Calculate and write in file the Berry curvature"""
  if operator is None: sz = operators.get_sz(h) # get sz operator
  else: sz = operator # assign operator
  be = [operator_berry(h,k=k,operator=sz,delta=delta) for k in kpath] 
  fo = open("BERRY_CURVATURE_SZ.OUT","w") # open file
  for (k,b) in zip(kpath,be):
    fo.write(str(k[0])+"   ")
    fo.write(str(k[1])+"   ")
    fo.write(str(b)+"\n")
  fo.close() # close file






def precise_spin_chern(h,delta=0.00001,tol=0.1):
  """ Calculates the chern number of a 2d system """
  from scipy import integrate
  err = {"epsabs" : 0.01, "epsrel": 0.01,"limit" : 20}
  sz = operators.get_sz(h) # get sz operator
  def f(x,y): # function to integrate
    return operator_berry(h,np.array([x,y]),delta=delta,operator=sz)
  c = integrate.dblquad(f,0.,1.,lambda x : 0., lambda x: 1.,epsabs=tol,
                          epsrel=tol)
  return c[0]/(2.*np.pi)


from .topologytk.green import berry_green_generator
from .topologytk.green import berry_green
from .topologytk.green import berry_operator



def berry_green_map_kpoint(h,emin=None,k=[0.,0.,0.],
        ne=100,dk=0.0001,operator=None,integral_mode="complex",
                  delta=0.002,integral=True,eps=1e-1,
                  energy=0.0,emax=0.0):
  """Return the Berry curvature map at a certain kpoint"""
  f = h.get_gk_gen(delta=delta,canonical_phase=True) # green function generator
  fgreen = berry_green_generator(f,k=k,dk=dk,operator=operator,full=True) 
  # No minimum energy provided
  if emin is None and integral:
      emin = algebra.eigvalsh(h.get_hk_gen()(k))[0] - 1.0
      print("Minimum energy",emin)
  def fint(x):  
#    return fgreen(x).trace()[0,0] # return diagonal
    return np.diag(fgreen(x)) # return diagonal
  ### The original function is defined in the complex plane,
  # we will do a change of variables of the form z = re^(iphi) - r0
  # so that dz = re^(iphi) i dphi
  if integral: # integrate up to the fermi energy
    es = np.linspace(emin,0.,ne) # energies used for the integration
    def fint2(x):
      """Function to integrate using a complex contour, from 0 to 1"""
      de = emax-emin # energy window of the integration
      ce = de/2. # center of the circle
      z0 = -ce*np.exp(-1j*x*np.pi) # parametrize the circle
      z = z0 + (emin+emax)/2. # shift the circle
      print("Evaluating",x)
      return -(fint(z)*z0).imag*np.pi # integral after the change of variables
    from .integration import integrate_matrix
    out = integrate_matrix(fint2,xlim=[0.,1.],eps=eps)
    out = out.real # turn real
  else: # evaluate at the fermi energy
    out = fint(energy).real
  return out # return result


def spatial_berry_density(h,**kwargs):
    return berry_green_map(h,integral=False,**kwargs)



def berry_green_map(h,nrep=5,k=[0.,0.,0.],operator=None,nk=None,**kwargs):
  """
  Write the Berry curvature of a kpoint in a file
  """
  if operator is not None: # this is a dirty workaround
    if type(operator) is str:
      operator = h.get_operator(operator) 
    else: pass
  if nk is None: # kpoint given
    out = berry_green_map_kpoint(h,k=k,operator=operator,**kwargs) 
  else: # kpoint not given
    if operator is not None: 
        from . import gauge
        operator = gauge.Operator2canonical_gauge(h,operator)
        print("Fixing the gauge in the operator")
    ks = klist.kmesh(h.dimensionality,nk=nk) # kpoints
    def f(ki):
        print("kpoint",ki)
        return berry_green_map_kpoint(h,k=ki,operator=operator,**kwargs)
    out = parallel.pcall(f,ks) # compute all
    out = np.mean(out,axis=0) # resum
  from . import geometry
  from .ldos import spatial_dos
  # write in a file
  geometry.write_profile(h.geometry,
          spatial_dos(h,out),name="BERRY_MAP.OUT",nrep=nrep)
  return out



def berry_density(h,k=[0.,0.,0.],operator=None,delta=0.02,dk=0.02):
  """Compute the Berry density"""
  f = h.get_gk_gen(delta=delta,canonical_phase=True) # green function generator
  fgreen = berry_green_generator(f,k=k,dk=dk,operator=operator,full=False) 
  return fgreen(0.0).real


def berry_density_map(h,nk=40,reciprocal=True,nsuper=1,
               delta=None,operator=None,dk=0.01):
  """Compute a Berry density map"""
  if delta is None: delta = 5./nk
  if reciprocal: R = h.geometry.get_k2K()
  else: R = np.matrix(np.identity(3))
  fo = open("BERRY_DENSITY_MAP.OUT","w") # open file
  nt = nk*nk # total number of points
  ik = 0
  ks = [] # list with kpoints
  for x in np.linspace(-nsuper,nsuper,nk,endpoint=False):
    for y in np.linspace(-nsuper,nsuper,nk,endpoint=False):
        ks.append([x,y,0.])
  tr = timing.Testimator("BERRY DENSITY",maxite=len(ks))
  def fp(ki): # function to compute the Berry curvature
      if parallel.cores == 1: tr.iterate()
      else: print("Doing",ki)
      r = np.matrix(ki).T # real space vectors
      k = np.array((R*r).T)[0] # change of basis
      b = berry_density(h,k=k,operator=operator,dk=dk) # get the density
      return b
  bs = parallel.pcall(fp,ks) # compute all the Berry curvatures
  for (b,k) in zip(bs,ks): # write everything
      fo.write(str(k[0])+"   "+str(k[1])+"     "+str(b)+"\n")
      fo.flush()
  fo.close() # close file



def chern_density(h,nk=10,operator=None,delta=0.02,dk=0.02,
        es=np.linspace(-1.0,1.0,40)):
  """Compute the Chern density as a function of the energy"""
  ks = klist.kmesh(h.dimensionality,nk=nk)
  cs = np.zeros(es.shape[0]) # initialize
  f = h.get_gk_gen(delta=delta,canonical_phase=True) # green function generator
  tr = timing.Testimator("CHERN DENSITY",maxite=len(ks))
  from . import parallel
  def fp(k): # compute berry curvatures
    if parallel.cores==1: tr.iterate()
    else: print(k)
#    k = np.random.random(3)
    fgreen = berry_green_generator(f,k=k,dk=dk,operator=operator,full=False) 
    return np.array([fgreen(e).real for e in es])
  out = parallel.pcall(fp,ks) # compute everything
  for o in out: cs += o # add contributions
  cs = cs/(len(ks)*np.pi*2) # normalize
  from scipy.integrate import cumtrapz
  csi = cumtrapz(cs,x=es,initial=0) # integrate
  np.savetxt("CHERN_DENSITY.OUT",np.matrix([es,cs]).T)
  np.savetxt("CHERN_DENSITY_INTEGRATED.OUT",np.matrix([es,csi]).T)








hall_conductivity = chern




def get_operator(h,op):
    """ Wrapper for operators """
    if op is None: return None
    if type(op)==str: # string
        if op=="valley": return h.get_operator("valley",projector=True) 
        else: return h.get_operator("valley",projector=True) 
    if callable(op): return op # function
    if type(op)==np.array: return op



from .topologytk import realspace

real_space_chern = realspace.real_space_chern

