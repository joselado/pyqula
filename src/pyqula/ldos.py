from __future__ import print_function
import scipy.sparse.linalg as slg
import scipy.linalg as lg
from scipy.sparse import csc_matrix as csc
from scipy.sparse import csc_matrix 
from scipy.sparse import bmat
import os
from numba import jit
import numpy as np
from . import klist
from . import operators
from . import timing
from . import parallel
from . import algebra
from .increase_hilbert import full2profile as spatial_dos
from . import filesystem as fs

def ldos0d(h,e=0.0,delta=0.01,write=True):
  """Calculates the local density of states of a Hamiltonian and
     writes it in file"""
  if h.dimensionality==0:  # only for 0d
    iden = np.identity(h.intra.shape[0],dtype=np.complex) # create identity
    g = ( (e+1j*delta)*iden -h.intra ).I # calculate green function
  else: raise # not implemented...
  d = [ -(g[i,i]).imag/np.pi for i in range(len(g))] # get imaginary part
  d = spatial_dos(h,d) # convert to spatial resolved DOS
  g = h.geometry  # store geometry
  if write: write_ldos(g.x,g.y,d,z=g.z) # write in file
  return d



def dos_site_kpm(h,energies=np.linspace(-1.,1.,1000),
        delta=0.01,scale = 10.0,i=0,nk=5,sector=None):
    """Compute a local DOS using the KPM"""
    h = h.copy()
    h.turn_sparse()
    npol = 5*int(scale/delta) # number of polynomials
    from . import kpm
    def getm(m,j): # for the matrix
      es,ds = kpm.ldos(m,i=j,scale=scale,npol=npol,ne=npol*5)
      return es,ds # return
    def get(j): # for the kpoint
        ks = klist.kmesh(h.dimensionality,nk=nk)
        hk = h.get_hk_gen() # get generator
        out = [getm(hk(k),j) for k in ks] # compute all
        ds = np.mean([o[1] for o in out],axis=0)
        es = out[0][0]
        return es,ds
#        if h.dimensonality!=0: raise # not implemented
#        return getm(h.intra)
    from scipy.interpolate import interp1d
    if h.has_spin and h.has_eh: # spin with electron-hole
        if sector is None:
            (es,ds1) = get(4*i)
            (es,ds2) = get(4*i+1)
            (es,ds3) = get(4*i+2)
            (es,ds4) = get(4*i+3)
            ds = ds1+ds2+ds3+ds4
        elif sector=="electron":
            (es,ds1) = get(4*i)
            (es,ds2) = get(4*i+1)
            ds = ds1+ds2
        else: raise
    elif h.has_spin and not h.has_eh: # spinful
        (es,ds1) = get(2*i)
        (es,ds2) = get(2*i+1)
        ds = ds1+ds2
    elif not h.has_spin and not h.has_eh: # spinless
        (es,ds) = get(i)
    else: raise
    f = interp1d(es,ds.real,bounds_error=False,fill_value=0.0)
    return energies,f(energies)





def dos_site(h,i=0,mode="ED",energies=np.linspace(-1.,1.,500),**kwargs):
    """DOS in a particular site for different energies"""
    if mode=="ED":
      if h.dimensionality!=0: raise # only for 0d
      out = []
      for e in energies:
          d = ldos0d(h,e=e,write=False,**kwargs)
          out.append(d[i]) # store
      return (energies,np.array(out)) # return result
    elif mode=="KPM":
        return dos_site_kpm(h,energies=energies,i=i,**kwargs)






def ldos0d_wf(h,e=0.0,delta=0.01,num_wf = 10,robust=False,tol=0):
  """Calculates the local density of states of a hamiltonian and
     writes it in file, using arpack"""
  if h.dimensionality==0:  # only for 0d
    intra = csc_matrix(h.intra) # matrix
  else: raise # not implemented...
  if robust: # go to the imaginary axis for stability
    eig,eigvec = slg.eigs(intra,k=int(num_wf),which="LM",
                        sigma=e+1j*delta,tol=tol) 
    eig = eig.real # real part only
  else: # Hermitic Hamiltonian
    eig,eigvec = slg.eigsh(intra,k=int(num_wf),which="LM",sigma=e,tol=tol) 
  d = np.array([0.0 for i in range(intra.shape[0])]) # initialize
  for (v,ie) in zip(eigvec.transpose(),eig): # loop over wavefunctions
    v2 = (np.conjugate(v)*v).real # square of wavefunction
    fac = delta/((e-ie)**2 + delta**2) # factor to create a delta
    d += fac*v2 # add contribution
#  d /= num_wf # normalize
  d /= np.pi # normalize
  d = spatial_dos(h,d) # resum if necessary
  g = h.geometry  # store geometry
  write_ldos(g.x,g.y,d,z=g.z) # write in file




def ldos_arpack(intra,num_wf=10,robust=False,tol=0,e=0.0,delta=0.01):
  """Use arpack to calculate hte local density of states at a certain energy"""
  if robust: # go to the imaginary axis for stability
    eig,eigvec = slg.eigs(intra,k=int(num_wf),which="LM",
                        sigma=e+1j*delta,tol=tol) 
    eig = eig.real # real part only
  else: # Hermitic Hamiltonian
    eig,eigvec = slg.eigsh(intra,k=int(num_wf),which="LM",sigma=e,tol=tol) 
  d = np.array([0.0 for i in range(intra.shape[0])]) # initialize
  for (v,ie) in zip(eigvec.transpose(),eig): # loop over wavefunctions
    v2 = (np.conjugate(v)*v).real # square of wavefunction
    fac = delta/((e-ie)**2 + delta**2) # factor to create a delta
    d += fac*v2 # add contribution
#  d /= num_wf # normalize
  d /= np.pi # normalize
  return d



def ldos_waves(intra,es = [0.0],delta=0.01,operator=None,
        num_bands=None,k=None,delta_discard=None):
  """Calculate the DOS in a set of energies by full diagonalization"""
  es = np.array(es) # array with energies
  if num_bands is None:
      eig,eigvec = algebra.eigh(intra) 
  else:
      eig,eigvec = algebra.smalleig(intra,numw=num_bands,evecs=True)
      eigvec = eigvec.T
  ds = [] # empty list
  if operator is None: weights = eig*0. + 1.0
  else: weights = [operator.braket(v,k=k) for v in eigvec.transpose()] # weights
  if delta_discard is not None: # discard too far values
      ewin = [min(es)-delta_discard*delta,min(es)+delta_discard*delta]
      for i in range(len(weights)):
          e = eig[i]
          if not ewin[0]<e<ewin[1]: weights[i] = 0.0
  v2s = [(np.conjugate(v)*v).real for v in eigvec.transpose()]
  ds = [[0.0 for i in range(intra.shape[0])] for e in es] # initialize
  ds = ldos_waves_jit(np.array(es),
          np.array(eigvec).T,np.array(eig),np.array(weights),
          np.array(v2s),np.array(ds),delta)
  return ds



@jit(nopython=True)
def ldos_waves_jit(es,eigvec,eig,weights,v2s,ds,delta):
  for i in range(len(es)): # loop over energies
    energy = es[i] # energy
    d = ds[i]
    for j in range(len(eig)):
        v = eigvec[j]
        ie = eig[j]
        weight = weights[j]
        v2 = v2s[j]
        fac = delta/((energy-ie)**2 + delta**2) # factor to create a delta
        d += weight*fac*v2 # add contribution
    d /= np.pi # normalize
    ds[i] = d # store
  return ds


def ldos_diagonalization(m,e=0.0,**kwargs):
    """Compute the LDOS using exact diagonalization"""
#    if algebra.issparse(m): return ldos_arpack(m,e=e,**kwargs) # sparse
    return ldos_waves(m,es=[e],**kwargs)[0] # dense




def ldosmap(h,energies=np.linspace(-1.0,1.0,40),delta=None,
        nk=40,operator=None,**kwargs):
  """Write a map of the ldos using full diagonalization"""
  if delta is None:
    delta = (np.max(energies)-np.min(energies))/len(energies) # delta
  hkgen = h.get_hk_gen() # get generator
  dstot = np.zeros((len(energies),h.intra.shape[0])) # initialize
  def getd(k): # get LDOS
    hk = hkgen(k) # get Hamiltonian
    # LDOS for this kpoint
    ds = ldos_waves(hk,k=k,es=energies,delta=delta,operator=operator,**kwargs) 
    return ds
  ks = [np.random.random(3) for ik in range(nk)] # kpoints
  ds = parallel.pcall(getd,ks) # get densities
  dstot = np.mean(ds,axis=0) # average over first axis
  print("LDOS finished")
  dstot = [spatial_dos(h,d) for d in dstot] # convert to spatial resolved DOS
  return energies,np.array(dstot)



def spatial_energy_profile(h,**kwargs):
  """Computes the DOS for each site of an slab, only for 2d"""
  if h.dimensionality==0:
      pos = h.geometry.x
      nk = 1
  elif h.dimensionality==1: pos = h.geometry.y
  elif h.dimensionality==2: pos = h.geometry.z
  else: raise
  es,ds = ldosmap(h,**kwargs)
  if len(ds[0])!=len(pos): 
    print("Wrong dimensions",len(ds[0]),len(pos))
    raise
  f = open("DOSMAP.OUT","w")
  f.write("# energy, index, DOS, position\n")
  for ie in range(len(es)):
    for ip in range(len(pos)):
      f.write(str(es[ie])+"  ")
      f.write(str(ip)+"  ")
      f.write(str(ds[ie,ip])+"   ")
      f.write(str(pos[ip])+"\n")
  f.close()
  return es,np.transpose(ds) # retunr LDOS 



slabldos = spatial_energy_profile # redefine





def ldos1d(h,e=0.0,delta=0.001,nrep=3):
  """ Calculate DOS for a 1d system"""
  from . import green
  if h.dimensionality!=1: raise # only for 1d
  gb,gs = green.green_renormalization(h.intra,h.inter,energy=e,delta=delta)
  d = [ -(gb[i,i]).imag for i in range(len(gb))] # get imaginary part
  d = spatial_dos(h,d) # convert to spatial resolved DOS
  g = h.geometry  # store geometry
  x,y = g.x,g.y # get the coordinates
  go = h.geometry.copy() # copy geometry
  go = go.supercell(nrep) # create supercell
  write_ldos(go.x,go.y,d.tolist()*nrep) # write in file
  return d


def ldos_projector(h,e=0.0,**kwargs):
    """Return an operator to project onto that region"""
    (x,y,d) = ldos(h,e=e,mode="arpack",silent=True,write=False,**kwargs)
    inds = np.array(range(len(d))) # indexes
    n = len(d)
    d = d/np.sum(d) # normalize
    m = csc_matrix((d,(inds,inds)),shape=(n,n),dtype=np.complex) # matrix
    m = h.spinless2full(m) # to full matrix
    return operators.Operator(m) # convert to operator

def ldos_density(h,**kwargs):
    """Return a normalized profile with the DOS"""
    (x,y,d) = ldos(h,mode="arpack",silent=True,write=False,**kwargs)
    d = d/np.sum(d) # normalize
    return d


def ldos_potential(h,**kwargs):
    """Return a function that evaluates an LDOS profile"""
    return # not finished yet




def get_ldos(h,projection="TB",**kwargs):
    """ Calculate LDOS"""
    if projection=="TB": return get_ldos_tb(h,**kwargs)
    elif projection=="atomic": 
        from .ldostk import atomicmultildos
        return atomicmultildos.get_ldos(h,**kwargs)
    else: raise


def get_ldos_tb(h,e=0.0,delta=0.001,nrep=5,nk=None,ks=None,mode="arpack",
             random=True,silent=False,interpolate=False,
             write=True,**kwargs):
  """ Calculate LDOS in a tight binding basis"""
  if ks is not None and mode=="green": raise
  if mode=="green":
    from . import green
    if h.dimensionality!=2: raise # only for 2d
    h = h.copy()
    h.turn_dense()
    if nk is not None:
      print("LDOS using normal integration with nkpoints",nk)
      gb,gs = green.bloch_selfenergy(h,energy=e,delta=delta,mode="full",nk=nk)
      d = [ -(gb[i,i]).imag for i in range(len(gb))] # get imaginary part
    else:
      print("LDOS using renormalization adaptative Green function")
      gb,gs = green.bloch_selfenergy(h,energy=e,delta=delta,mode="adaptive")
      d = [ -(gb[i,i]).imag for i in range(len(gb))] # get imaginary part
  elif mode=="arpack" or mode=="diagonalization": # arpack diagonalization
    from . import klist
    if nk is None: nk = 10
    hkgen = h.get_hk_gen() # get generator
    ds = [] # empty list
    if ks is None:
      ks = klist.kmesh(h.dimensionality,nk=nk)
      if random: ks = [np.random.random(3) for k in ks] # random mesh
    ts = timing.Testimator(title="LDOS",maxite=len(ks),silent=silent)
    for k in ks: # loop over kpoints
      ts.iterate()
      hk = hkgen(k) # get Hamiltonian
      ds += [ldos_diagonalization(hk,e=e,delta=delta,**kwargs)]
    d = np.mean(ds,axis=0) # average
  else: raise # not recognized
  # write result
  d = spatial_dos(h,d) # convert to spatial resolved DOS
  g = h.geometry  # store geometry
  x,y = g.x,g.y # get the coordinates
  if write: 
      from .interpolation import atomic_interpolation
      go = h.geometry.copy() # copy geometry
      go = go.supercell(nrep) # create supercell
      do = d.tolist()*(nrep**g.dimensionality) # replicate
      xo = go.x
      yo = go.y
      if interpolate:
        xo,yo,do = atomic_interpolation(xo,yo,do)
      write_ldos(xo,yo,do) # write in file
  return (x,y,d) # return LDOS


ldos = get_ldos # for backcompatibility

def multi_ldos(h,projection="TB",**kwargs):
    """Compute the LDOS at different energies, and save everything in a file"""
    if projection=="TB": return multi_ldos_tb(h,**kwargs)
    elif projection=="atomic": 
        from .ldostk import atomicmultildos
        return atomicmultildos.multi_ldos(h,**kwargs)


def multi_ldos_tb(h,es=np.linspace(-1.0,1.0,100),delta=0.01,
        nrep=3,nk=100,num_bands=20,
        random=False,op=None,**kwargs):
  """Calculate many LDOS, by diagonalizing the Hamiltonian"""
  print("Calculating eigenvectors in LDOS")
  ps = [] # weights
  evals,ws = [],[] # empty list
  ks = klist.kmesh(h.dimensionality,nk=nk) # get grid
  hk = h.get_hk_gen() # get generator
  op = operators.tofunction(op) # turn into a function
#  if op is None: op = lambda x,k: 1.0 # dummy function
  if h.is_sparse: # sparse Hamiltonian
    from .bandstructure import smalleig
    print("SPARSE Matrix")
    for k in ks: # loop
      print("Diagonalizing in LDOS, SPARSE mode")
      if random:
        k = np.random.random(3) # random vector
        print("RANDOM vector in LDOS")
      e,w = smalleig(hk(k),numw=num_bands,evecs=True)
      evals += [ie for ie in e]
      ws += [iw for iw in w]
      ps += [op(iw,k=k) for iw in w] # weights
  else:
    print("Diagonalizing in LDOS, DENSE mode")
    for k in ks: # loop
      if random:
        k = np.random.random(3) # random vector
        print("RANDOM vector in LDOS")
      e,w = algebra.eigh(hk(k))
      w = w.transpose()
      evals += [ie for ie in e]
      ws += [iw for iw in w]
      ps += [op(iw,k=k[0]) for iw in w] # weights
#      evals = np.concatenate([evals,e]) # store
#      ws = np.concatenate([ws,w]) # store
  ds = [(np.conjugate(v)*v).real for v in ws] # calculate densities
  del ws # remove the wavefunctions
  fs.rmdir("MULTILDOS") # remove folder
  fs.mkdir("MULTILDOS") # create folder
  go = h.geometry.copy() # copy geometry
  go = go.supercell(nrep) # create supercell
  fo = open("MULTILDOS/MULTILDOS.TXT","w") # files with the names
  def getldosi(e):
    """Get this iteration"""
    out = np.array([0.0 for i in range(h.intra.shape[0])]) # initialize
    for (d,p,ie) in zip(ds,ps,evals): # loop over wavefunctions
      fac = delta/((e-ie)**2 + delta**2) # factor to create a delta
      out += fac*d*p # add contribution
    out /= np.pi # normalize
    return spatial_dos(h,out) # resum if necessary
  outs = parallel.pcall(getldosi,es) # get energies
  ie = 0
  for e in es: # loop over energies
    print("MULTILDOS for energy",e)
    out = outs[ie] ; ie += 1 # get and increase
    name0 = "LDOS_"+str(e)+"_.OUT" # name of the output
    name = "MULTILDOS/" + name0
    write_ldos(go.x,go.y,out.tolist()*(nrep**h.dimensionality),
                  output_file=name) # write in file
    fo.write(name0+"\n") # name of the file
    fo.flush() # flush
  fo.close() # close file
  fmap = open("DOSMAP.OUT","w")
  for ii in range(len(h.geometry.x)):
      for ie in range(len(es)):
          fmap.write(str(ii)+"  ")
          fmap.write(str(es[ie])+"  ")
          fmap.write(str(outs[ie][ii])+"\n")
  fmap.close()
  # Now calculate the DOS
  from .dos import calculate_dos
  es2 = np.linspace(min(es),max(es),len(es)*10)
  ys = calculate_dos(evals,es2,delta,w=None) # compute DOS
  from .dos import write_dos
  write_dos(es2,ys,output_file="MULTILDOS/DOS.OUT")  





def write_ldos(x,y,dos,output_file="LDOS.OUT",z=None):
  """ Write LDOS in a file"""
  fd = open(output_file,"w")   # open file
  fd.write("# x,  y, local density of states\n")
  ii = 0
  for (ix,iy,idos) in zip(x,y,dos): # write everything
    fd.write(str(ix) +"   "+ str(iy) + "   "+ str(idos))
    if z is not None: fd.write("   "+str(z[ii]))
    fd.write("\n")
    ii += 1
  fd.close() # close file




def ldos_finite(h,e=0.0,n=10,nwf=4,delta=0.0001):
  """Calculate the density of states for a finite system"""
  if h.dimensionality!=1: raise # if it is not one dimensional
  intra = csc(h.intra) # convert to sparse
  inter = csc(h.inter) # convert to sparse
  interH = inter.H # hermitian
  m = [[None for i in range(n)] for j in range(n)] # full matrix
  for i in range(n): # add intracell
    m[i][i] = intra
  for i in range(n-1): # add intercell
    m[i][i+1] = inter
    m[i+1][i] = interH
  m = bmat(m) # convert to matrix
  (ene,wfs) = slg.eigsh(m,k=nwf,which="LM",sigma=0.0) # diagonalize
  wfs = wfs.transpose() # transpose wavefunctions
  dos = (wfs[0].real)*0.0 # calculate dos
  for (ie,f) in zip(ene,wfs): # loop over waves
    c = 1./(1.+((ie-e)/delta)**2) # calculate coefficient
    dos += np.abs(f)*c # add contribution
  odos = spatial_dos(h,dos) # get the spatial distribution
  go = h.geometry.supercell(n) # get the supercell
  write_ldos(go.x,go.y,odos) # write in a file
  return dos # return the dos





def ldos_defect(h,v,e=0.0,delta=0.001,n=1):
  """Calculates the LDOS of a cell with a defect, writting the n
  neighring cells"""
  raise # still not finished
  from . import green
  # number of repetitions
  rep = 2*n +1
  # calculate pristine green function
  g,selfe = green.supercell_selfenergy(h,e=e,delta=delta,nk=100,nsuper=rep)
  # now calculate defective green function 
  ez = e + 1j*delta # complex energy
  emat = np.matrix(np.identity(len(g)))*ez  # E +i\delta 
  from . import supercell
  pintra = supercell.intra_super2d(h,n=rep) # pristine
  vintra = supercell.intra_super2d(h,n=rep,central=v) # defective
  selfe = emat - pintra - g.I # dyson euqation, get selfenergy
  gv = (emat - vintra -selfe).I   # Green function of a vacancy, with selfener
  return



from .ldostk.ldosr import ldosr_generator


