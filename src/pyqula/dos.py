from __future__ import print_function
import numpy as np
import scipy.linalg as lg
from .algebra import smalleig # arpack diagonalization
import time
from . import timing
from . import kpm
from . import checkclass
from . import green
from . import algebra
from . import parallel
from numba import jit
from .klist import kmesh

try:
#  raise
  from . import dosf90 
  use_fortran = True
except:
#  print("Something wrong with FORTRAN in DOS")
  use_fortran = False


def calculate_dos(es,xs,d,use_fortran=use_fortran,w=None):
  if w is None: w = np.zeros(len(es)) + 1.0 # initialize
  if use_fortran: # use fortran routine
    from . import dosf90 
    return dosf90.calculate_dos(es,xs,d,w) # use the Fortran routine
  else:
      ys = np.zeros(xs.shape[0]) # initialize
      ys = calculate_dos_jit(es,xs,d,w,ys) # compute
      return ys

@jit
def calculate_dos_jit(es,xs,d,w,ys):
      for i in range(len(es)): # loop over energies
          e = es[i]
          iw = w[i]
          de = xs - e # E - Ei
          de = d/(d*d + de*de) # 1/(delta^2 + (E-Ei)^2)
          ys += de*iw # times the weight
      return ys




def dos_surface(h,output_file="DOS.OUT",
                 energies=np.linspace(-1.,1.,20),delta=0.001):
  """Calculates the DOS of a surface, and writes in file"""
  if h.dimensionality!=1: raise # only for 1d
  fo = open(output_file,"w")
  fo.write("# energy, DOS surface, DOS bulk\n")
  for e in energies: # loop over energies
    print("Done",e)
    gb,gs = green.green_renormalization(h.intra,h.inter,energy=e,delta=delta)
    gb = -gb.trace()[0,0].imag
    gs = -gs.trace()[0,0].imag
    fo.write(str(e)+"     "+str(gs)+"    "+str(gb)+"\n")
  fo.close()




def dos0d(h,energies=np.linspace(-4,4,500),delta=0.01):
  """Calculate density of states of a 0d system"""
  hkgen = h.get_hk_gen() # get generator
  calculate_dos_hkgen(hkgen,[0],
            delta=delta,energies=energies) # conventiona algorithm
#  ds = [] # empty list
#  if h.dimensionality==0:  # only for 0d
#    iden = np.identity(h.intra.shape[0],dtype=np.complex) # create identity
#    for e in es: # loop over energies
#      g = ( (e+1j*delta)*iden -h.intra ).I # calculate green function
#      if i is None: d = -g.trace()[0,0].imag
#      elif checkclass.is_iterable(i): # iterable list 
#          d = sum([-g[ii,ii].imag for ii in i])
#      else: d = -g[i,i].imag # assume an integer
#      ds.append(d)  # add this dos
#  else: raise # not implemented...
#  write_dos(es,ds)
#  return ds





def dos0d_kpm(h,use_kpm=True,scale=10,npol=100,ntries=100,fun=None):
  """ Calculate density of states of a 1d system"""
  if h.dimensionality!=0: raise # only for 0d
  if not use_kpm: raise # only using KPM
  h.turn_sparse() # turn the hamiltonian sparse
  mus = np.array([0.0j for i in range(2*npol)]) # initialize polynomials
  mus = kpm.random_trace(h.intra/scale,ntries=ntries,n=npol,fun=fun)
  xs = np.linspace(-0.9,0.9,4*npol) # x points
  ys = kpm.generate_profile(mus,xs) # generate the profile
  write_dos(xs*scale,ys) # write in file


def dos0d_sites(h,sites=[0],scale=10.,npol=500,ewindow=None,refine_e=1.0):
  """ Calculate density of states of a 1d system for a certain orbitals"""
  if h.dimensionality!=0: raise # only for 1d
  h.turn_sparse() # turn the hamiltonian sparse
  mus = np.array([0.0j for i in range(2*npol)]) # initialize polynomials
  hk = h.intra # hamiltonian
  for isite in sites:
    mus += kpm.local_dos(hk/scale,i=isite,n=npol)
  if ewindow is None:  xs = np.linspace(-0.9,0.9,int(npol*refine_e)) # x points
  else:  xs = np.linspace(-ewindow/scale,ewindow/scale,npol) # x points
  ys = kpm.generate_profile(mus,xs) # generate the profile
  write_dos(xs*scale,ys) # write in file











def write_dos(es,ds,output_file="DOS.OUT"):
  """ Write DOS in a file"""
  f = open(output_file,"w")
  for (e,d) in zip(es,ds):
    f.write(str(e)+"     ")
    f.write(str(d.real)+"\n")
  f.close()




def dos1d(h,use_kpm=False,scale=10.,nk=100,npol=100,ntries=2,
          ndos=1000,delta=0.01,ewindow=None,frand=None,
          energies=None):
  """ Calculate density of states of a 1d system"""
  if h.dimensionality!=1: raise # only for 1d
  ks = np.linspace(0.,1.,nk,endpoint=False) # number of kpoints
  if not use_kpm: # conventional method
    hkgen = h.get_hk_gen() # get generator
#    delta = 16./(nk*h.intra.shape[0]) # smoothing
    calculate_dos_hkgen(hkgen,ks,
            delta=delta,energies=energies) # conventiona algorithm
    return np.genfromtxt("DOS.OUT").transpose()
  else:
    h.turn_sparse() # turn the hamiltonian sparse
    hkgen = h.get_hk_gen() # get generator
    yt = np.zeros(ndos) # number of dos
    ts = timing.Testimator("DOS") 
    for i in range(nk): # loop over kpoints
      k = np.random.random(3) # random k-point
      hk = hkgen(k) # hamiltonian
      (xs,yi) = kpm.tdos(hk,scale=scale,npol=npol,frand=frand,ewindow=ewindow,
                  ntries=ntries,ne=ndos)
      yt += yi # Add contribution
      ts.remaining(i,nk)
    yt /= nk # normalize
    write_dos(xs,yt) # write in file
    return xs,yt




def dos1d_sites(h,sites=[0],scale=10.,nk=100,npol=100,info=False,ewindow=None):
  """ Calculate density of states of a 1d system for a certain orbitals"""
  if h.dimensionality!=1: raise # only for 1d
  ks = np.linspace(0.,1.,nk,endpoint=False) # number of kpoints
  h.turn_sparse() # turn the hamiltonian sparse
  hkgen = h.get_hk_gen() # get generator
  mus = np.array([0.0j for i in range(2*npol)]) # initialize polynomials
  for k in ks: # loop over kpoints
    hk = hkgen(k) # hamiltonian
    for isite in sites:
      mus += kpm.local_dos(hk/scale,i=isite,n=npol)
    if info: print("Done",k)
  mus /= nk # normalize by the number of kpoints
  if ewindow is None:  xs = np.linspace(-0.9,0.9,npol) # x points
  else:  xs = np.linspace(-ewindow/scale,ewindow/scale,npol) # x points
  ys = kpm.generate_profile(mus,xs) # generate the profile
  write_dos(xs*scale,ys) # write in file


def calculate_dos_hkgen(hkgen,ks,ndos=100,delta=None,
         is_sparse=False,numw=10,window=None,energies=None):
  """Calculate density of states using the ks given on input"""
  if not is_sparse: # if not is sparse
      m = hkgen([0,0,0]) # get the matrix
      if algebra.issparse(m): 
          print("Hamiltonian is sparse, selecting sparse mode in DOS")
          is_sparse = True
  es = np.zeros((len(ks),hkgen(ks[0]).shape[0])) # empty list
  tr = timing.Testimator("DOS",maxite=len(ks))
  if delta is None: delta = 5./len(ks) # automatic delta
  from . import parallel
  def fun(k): # function to execute
    if parallel.cores==1: tr.iterate() # print the info
    hk = hkgen(k) # Hamiltonian
    t0 = time.perf_counter() # time
    if is_sparse: # sparse Hamiltonian 
      es = algebra.smalleig(hk,numw=numw,tol=delta/1e3) # eigenvalues
      ws = np.zeros(es.shape[0])+1.0 # weight
    else: # dense Hamiltonian
      es = algebra.eigvalsh(hk) # get eigenvalues
      ws = np.zeros(es.shape[0])+1.0 # weight
    return es # return energies
#  for ik in range(len(ks)):  
  out = parallel.pcall(fun,ks) # launch all the processes
  es = [] # empty list
  for o in out: 
      es = np.concatenate([es,o]) # concatenate
#    tr.remaining(ik,len(ks))
#  es = es.reshape(len(es)*len(es[0])) # 1d array
  es = np.array(es) # convert to array
  nk = len(ks) # number of kpoints
  if energies is not None: # energies given on input
    xs = energies
  else:
    if window is None:
      xs = np.linspace(np.min(es)-.5,np.max(es)+.5,ndos) # create x
    else:
      xs = np.linspace(-window,window,ndos) # create x
  ys = calculate_dos(es,xs,delta) # use the Fortran routine
  ys /= nk # normalize by the number of k-points
  ys *= 1./np.pi # normalization of the Lorentzian
  write_dos(xs,ys) # write in file
  print("\nDOS finished")
  return (xs,ys) # return result



def dos_kmesh(h,nk=10,delta=1e-3,random=False,ks=None,
        energies=np.linspace(-1,1,200),**kwargs):
    """Compute the DOS in a k-mesh by using the bandstructure function"""
    if ks is None:  ks = kmesh(h.dimensionality,nk=nk)
    if random: ks = [np.random.random(3) for k in ks]
    # compute band structure
    out = h.get_bands(kpath=ks,write=False,**kwargs) 
    if len(out)==2: w = None
    else: w = out[2]
    ys = calculate_dos(out[1],energies,delta,w=w)/len(ks)
    ys *= 1./np.pi # normalization of the Lorentzian
    write_dos(energies,ys) # write in file
    print("\nDOS finished")
    return (energies,ys) # return result









def dos2d(h,use_kpm=False,scale=10.,nk=100,ntries=1,delta=None,
          ndos=2000,random=True,kpm_window=1.0,
          window=None,energies=None,**kwargs):
  """ Calculate density of states of a 2d system"""
  if h.dimensionality!=2: raise # only for 2d
  ks = []
  from .klist import kmesh
  ks = kmesh(h.dimensionality,nk=nk)
  if random:
    ks = [np.random.random(2) for ik in ks]
    print("Random k-mesh")
  if not use_kpm: # conventional method
    hkgen = h.get_hk_gen() # get generator
    if delta is None: delta = 6./nk
# conventiona algorithm
    (xs,ys) = calculate_dos_hkgen(hkgen,ks,ndos=ndos,delta=delta,
                          is_sparse=h.is_sparse,window=window,
                          energies=energies,**kwargs) 
    write_dos(xs,ys) # write in file
    return (xs,ys)
  else: # use the kpm
    if delta is not None: npol = int(20*scale/delta)
    else: npol = ndos//10
    h.turn_sparse() # turn the hamiltonian sparse
    hkgen = h.get_hk_gen() # get generator
    mus = np.array([0.0j for i in range(2*npol)]) # initialize polynomials
    tr = timing.Testimator("DOS")
    ik = 0
    from . import parallel
    if parallel.cores==1: # serial run
      for k in ks: # loop over kpoints
        ik += 1
        tr.remaining(ik,len(ks))      
        if random: 
          kr = np.random.random(2)
          print("Random sampling in DOS")
          hk = hkgen(kr) # hamiltonian
        else: hk = hkgen(k) # hamiltonian
        mus += kpm.random_trace(hk/scale,ntries=ntries,n=npol)
    else:
        ff = lambda k: kpm.random_trace(hkgen(k)/scale,ntries=ntries,n=npol)
        mus = parallel.pcall(ff,ks) # parallel call
        mus = np.array(mus).sum(axis=0) # sum all the contributions
    mus /= len(ks) # normalize by the number of kpoints
    if energies is None:
      xs = np.linspace(-0.9,0.9,ndos)*kpm_window # x points
    else:  xs = energies/scale
    ys = kpm.generate_profile(mus,xs) # generate the profile
    ys /= scale # rescale
    write_dos(xs*scale,ys) # write in file
    return (xs,ys)




def dos3d(h,scale=10.,nk=20,delta=None,ndos=100,
        random=False,energies=None):
  """ Calculate density of states of a 2d system"""
  if h.dimensionality!=3: raise # only for 2d
  ks = [np.random.random(3) for i in range(nk)] # number of kpoints
  hkgen = h.get_hk_gen() # get generator
  if delta is None: delta = 10./ndos # smoothing
  return calculate_dos_hkgen(hkgen,ks,ndos=ndos,delta=delta,energies=energies) 











def dos2d_ewindow(h,energies=np.linspace(-1.,1.,30),delta=None,info=False,
                    use_green=True,nk=300,mode="adaptive"):
  """Calculate the density of states in certain eenrgy window"""
  ys = [] # density of states
  if delta is None: # pick a good delta value
    delta = 0.1*(max(energies) - min(energies))/len(energies)
  if use_green:
    from .green import bloch_selfenergy
    for energy in energies:
      (g,selfe) = bloch_selfenergy(h,nk=nk,energy=energy, delta=delta,
                   mode=mode)
      ys.append(-g.trace()[0,0].imag)
      if info: print("Done",energy)
    write_dos(energies,ys) # write in file
    return
  else: # do not use green function    
    import scipy.linalg as lg
    kxs = np.linspace(0.,1.,nk)
    kys = np.linspace(0.,1.,nk)
    hkgen= h.get_hk_gen() # get hamiltonian generator
    ys = energies*0.
    weight = 1./(nk*nk)
    for ix in kxs:
      for iy in kys:
        k = np.array([ix,iy,0.]) # create kpoint
        hk = hkgen(k) # get hk hamiltonian
        evals = lg.eigvalsh(hk) # get eigenvalues
        ys += weight*calculate_dos(evals,energies,delta) # add this contribution
      if info: print("Done",ix)
    write_dos(energies,ys) # write in file
    return






def dos1d_ewindow(h,energies=np.linspace(-1.,1.,30),delta=None,info=False,
                    use_green=True,nk=300):
  """Calculate the density of states in certain energy window"""
  ys = [] # density of states
  if delta is None: # pick a good delta value
    delta = 0.1*(max(energies) - min(energies))/len(energies)
  if True: # do not use green function    
    import scipy.linalg as lg
    kxs = np.linspace(0.,1.,nk)
    hkgen= h.get_hk_gen() # get hamiltonian generator
    ys = energies*0.
    weight = 1./(nk)
    for ix in kxs:
      hk = hkgen(ix) # get hk hamiltonian
      evals = lg.eigvalsh(hk) # get eigenvalues
      ys += weight*calculate_dos(evals,energies,delta) # add this contribution
    if info: print("Done",ix)
    write_dos(energies,ys) # write in file
    return











def dos_ewindow(h,energies=np.linspace(-1.,1.,30),delta=None,info=False,
                    use_green=True,nk=300):
  """ Calculate density of states in an energy window"""
  if h.dimensionality==2: # two dimensional
    dos2d_ewindow(h,energies=energies,delta=delta,info=info,
                    use_green=use_green,nk=nk)
  elif h.dimensionality==1: # one dimensional
    dos1d_ewindow(h,energies=energies,delta=delta,info=info,
                    use_green=use_green,nk=nk)
  else: raise # not implemented






def convolve(x,y,delta=None):
  """Add a broadening to a DOS"""
  if delta is None: return y # do nothing
  delta = np.abs(delta) # absolute value
  xnew = np.linspace(-1.,1.,len(y)) # array
  d2 = delta/(np.max(x) - np.min(x)) # effective broadening
  fconv = d2/(xnew**2 + d2**2) # convolving function
  yout = np.convolve(y,fconv,mode="same") # same size
  # ensure the normaliation is the same
  ratio = np.sum(np.abs(y))/np.sum(np.abs(yout))
  yout *= ratio
  return yout # return new array





def dos_kpm(h,scale=10.0,ewindow=4.0,ne=10000,
        delta=0.01,ntries=10,nk=100,operator=None,
        random=True,
        **kwargs):
  """Calculate the KDOS bands using the KPM"""
  hkgen = h.get_hk_gen() # get generator
  ks = kmesh(h.dimensionality,nk=nk) # klist
  if random: ks = [np.random.random(3) for k in ks]
  ytot = np.zeros(ne) # initialize
  npol = 5*int(scale/delta) # number of polynomials
  def f(k):
    hk = hkgen(k) # get Hamiltonian
    if callable(operator): op = operator(k) # call the function if necessary
    else: op = operator # take the same operator
    (x,y) = kpm.tdos(hk,scale=scale,npol=npol,ne=ne,operator=op,
                   ewindow=ewindow,ntries=ntries,**kwargs) # compute
    return (x,y)
  from . import parallel
  numk = len(ks)
  if parallel.cores==1:
    tr = timing.Testimator("DOS",maxite=numk) # generate object
    for ik in range(len(ks)): # loop over kpoints
      tr.iterate()
      k = ks[ik]
      (x,y) = f(k) # compute
      ytot += y # add contribution
    ytot /= nk # normalize
  else: # parallel calculation
    out = parallel.pcall(f,ks) # compute all
    ytot = np.mean([out[i][1] for i in range(numk)],axis=0) # average DOS
    x = out[0][0] # energies
  np.savetxt("DOS.OUT",np.matrix([x,ytot]).T) # save in file
  return (x,ytot)



def get_dos(h,energies=np.linspace(-4.0,4.0,400),
            use_kpm=False,mode="ED",**kwargs):
  """Calculate the density of states"""
  if use_kpm: # KPM
      ewindow = max([abs(min(energies)),abs(min(energies))]) # window
      return dos_kpm(h,ewindow=ewindow,ne=len(energies),**kwargs)
  else: # conventional methods
      if mode=="ED": # exact diagonalization
          return dos_kmesh(h,energies=energies,**kwargs)
      elif mode=="Green": # Green function formalism
          def fun(e):
              return green.green_operator(h,e=e,**kwargs) 
          ds = parallel.pcall(fun,energies) # compute DOS with an operator
          np.savetxt("DOS.OUT",np.array([energies,ds]).T) # write in a file
          return (energies,ds)
      else: 
          print("Unrecognized option in DOS")
          raise


dos = get_dos # redefine

def autodos(h,auto=True,**kwargs):
    """Automatic computation of DOS"""
    if auto: # automatic integration
        def f(): return dos(h,**kwargs)[1] # return dos
        x = dos(h,**kwargs)[0] # energies
        from .integration import random_integrate
        y = random_integrate(f)
        np.savetxt("DOS.OUT",np.array([x,y]).T)
        return (x,y)
    return dos(h,**kwargs)




def bulkandsurface(h1,energies=np.linspace(-1.,1.,100),operator=None,
                    delta=0.01,hs=None,nk=30):
  """Compute the DOS of the bulk and the surface"""
  tr = timing.Testimator("KDOS") # generate object
  ik = 0
  h1 = h1.get_multicell() # multicell Hamiltonian
  kpath = [np.random.random(3) for i in range(nk)] # loop
  dosout = np.zeros((2,len(energies))) # output DOS
  for k in kpath:
    tr.remaining(ik,len(kpath)) # generate object
    ik += 1
    outs = green.surface_multienergy(h1,k=k,energies=energies,delta=delta,hs=hs)
    ie = 0
    for (energy,out) in zip(energies,outs): # loop over energies
      # compute dos
      ig = 0
      for g in out: # loop
        if operator is None: d = -g.trace()[0,0].imag # only the trace 
        elif callable(operator): d = operator(g,k=k) # call the operator
        else:  d = -(g*operator).trace()[0,0].imag # assume it is a matrix
        dosout[ig,ie] += d # store
        ig += 1 # increase site
      ie += 1 # increase energy
  # write in file
  dosout/=nk # normalize
  np.savetxt("DOS_BULK_SURFACE.OUT",np.matrix([energies,dosout[0],dosout[1]]).T)




def surface2bulk(h,n=50,nk=3000,delta=1e-3,e=0.0,**kwargs):
    """Compute the DOS from the surface to the bulk"""
    if h.dimensionality==2:
      def f(k):
        (ons,hop) = green.get1dhamiltonian(h,[k,0.,0.])
        gf,sf = green.green_renormalization(ons,hop,energy=e,
                delta=delta,**kwargs)
        out = green.green_surface_cells(sf,hop,ons,delta=delta,e=e,n=n)
        return np.array([-algebra.trace(o).imag for o in out]) # DOS
      ks = np.linspace(0.,1.,nk) # loop
      out = np.mean([f(k) for k in ks],axis=0)
    else: raise
    return np.array([range(n),out]) # return array





