from __future__ import print_function
import numpy as np
import scipy.linalg as lg
from . import multicell
from . import algebra
from .algebra import dagger
from numba import jit
from .greentk.rg import green_renormalization
from .greentk.selfenergy import bloch_selfenergy
from .greentk.kchain import green_kchain
from .greentk.kchain import get1dhamiltonian
from .greentk.kchain import green_kchain_evaluator


use_fortran = False

try: from .gauss_invf90 import gauss_inv as ginv
except: 
  pass


class gf_convergence():
   """ Class to manage the convergence  options
   of the green functions """
   optimal = False
   refinement = False
   guess = True # use old green function
   def __init__(self,mode):
     if mode=="fast":   # fast mode,used for coule to finite systems 
       self.eps = 0.001
       self.max_error = 1.0
       self.num_rep = 10
       self.mixing = 1.0
     if mode=="lead":
       self.eps = 0.001
       self.max_error = 0.00001
       self.num_rep = 3
       self.mixing = 0.8
     if mode=="hundred":  
       self.eps = 0.001
       self.max_error = 1.0
       self.num_rep = 100
       self.mixing = 1.0



def dyson(intra,inter,energy=0.0,gf=None,is_sparse=False,initial = None):
  """ Solves the dyson equation for a one dimensional
  system with intra matrix 'intra' and inter to the nerest cell
  'inter'"""
  # get parameters
  if gf is None: gf = gf_convergence("lead")
  mixing = gf.mixing
  eps = gf.eps
  max_error = gf.max_error
  num_rep = gf.num_rep
  optimal = gf.optimal
  try:
    intra = intra.todense()
    inter = inter.todense()
  except:
    a = 1
  if initial is None:  # if green not provided. initialize at zero
    from numpy import zeros
   
    g_guess = intra*0.0j
  else:
    g_guess = initial
  # calculate using fortran
  if optimal:
    print("Fortran dyson calculation")
    from .green_fortran import dyson  # import fortran subroutine
    (g,num_redo) = dyson(intra,inter,energy,num_rep,mixing=mixing,
               eps=eps,green_guess=g_guess,max_error=max_error)
    print("      Converged in ",num_redo,"iterations\n")
    from numpy import matrix
    g = matrix(g)
  # calculate using python
  if not optimal:
    g_old = g_guess # first iteration
    iden = np.matrix(np.identity(len(intra),dtype=complex)) # create identity
    e = iden*(energy+1j*eps) # complex energy
    while True: # loop over iterations
      self = inter@g_old@inter.H # selfenergy
      g = (e - intra - self).I # dyson equation
      if np.max(np.abs(g-g_old))<gf.max_error: break
      g_old = mixing*g + (1.-mixing)*g_old # new green function
  if is_sparse: 
    from scipy.sparse import csc_matrix
    g = csc_matrix(g)
  return g











def dos_infinite(intra,inter,energies=[0.0],num_rep=100,
                      mixing=0.7,eps=0.0001,green_guess=None,max_error=0.0001):
   """ Calculates the surface density of states by using a 
    green function approach"""
   dos = [] # list with the density of states
   iden = np.matrix(np.identity(len(intra),dtype=complex)) # create idntity
   for energy in energies: # loop over energies
     # right green function
     gr = dyson(intra,inter,energy=energy,num_rep=num_rep,mixing=mixing,
          eps=eps,green_guess=green_guess,max_error=max_error)
     # left green function
     gl = dyson(intra,inter.H,energy=energy,num_rep=num_rep,mixing=mixing,
          eps=eps,green_guess=green_guess,max_error=max_error)
     # central green function
     selfl = inter.H@gl@inter # left selfenergy
     selfr = inter@gr@inter.H # right selfenergy
     gc = energy*iden -intra -selfl -selfr # dyson equation for the center
     gc = gc.I # calculate inverse
     dos.append(-algebra.trace(gc).imag)  # calculate the trace of the Green function
   return dos




def dos_semiinfinite(intra,inter,energies=np.linspace(-1.0,1.0,100),num_rep=100,
                      mixing=0.7,eps=0.0001,green_guess=None,max_error=0.0001):
   """ Calculates the surface density of states by using a 
    green function approach"""
   dos = [] # list with the density of states
   for energy in energies: # loop over energies
#     gf = dyson(intra,inter,energy=energy,num_rep=num_rep,mixing=mixing,
     gb,gf = green_renormalization(intra,inter,energy=energy,delta=delta)
     dos.append(-algebra.trace(gf).imag)  # calculate the trace of the Green function
   return energies,dos









def dos_heterostructure(hetero,energies=[0.0],num_rep=100,
                      mixing=0.7,eps=0.0001,green_guess=None,max_error=0.0001):
   """ Calculates the density of states 
       of a heterostructure by a  
    green function approach, input is a heterostructure class"""
   dos = [] # list with the density of states
   iden = np.matrix(np.identity(len(intra),dtype=complex)) # create idntity
   for energy in energies: # loop over energies
     # right green function
     intra = hetero.right_intra
     inter = hetero.right_inter
     gr = dyson(intra,inter,energy=energy,num_rep=num_rep,mixing=mixing,
          eps=eps,green_guess=green_guess,max_error=max_error)
     # left green function
     intra = hetero.right_intra
     inter = hetero.right_inter
     gl = dyson(intra,inter,energy=energy,num_rep=num_rep,mixing=mixing,
          eps=eps,green_guess=green_guess,max_error=max_error)
     # central green function
     selfl = inter.H@gl@inter # left selfenergy
     selfr = inter@gr@inter.H # right selfenergy
     gc = energy*iden -intra -selfl -selfr # dyson equation for the center
     gc = gc.I # calculate inverse
     dos.append(-algebra.trace(gc).imag)  # calculate the trace of the Green function
   return dos



def read_matrix(f):
  """Read green function from a file"""
  m = np.genfromtxt(f)
  d = int(max(m.transpose()[0]))+1 # dimension of the green functions
  g = np.matrix([[0.0j for i in range(d)] for j in range(d)]) # create matrix
  for r in m:
    i = int(r[0])
    j = int(r[1])
    ar = r[2]
    ai = r[3]
    g[i,j] = ar +1j*ai # store element
  return g # return green function



def write_matrix(f,g):
  """Write green function from a file"""
  fw = open(f,"w") # open file to write
  n = len(g) # dimension of the matrix
  for i in range(n):
    for j in range(n):
      fw.write(str(i)+"  ")
      fw.write(str(j)+"  ")
      fw.write(str(g[i,j].real)+"  ")
      fw.write(str(g[i,j].imag)+"\n")
  fw.close()   # close file


# detect non vanishing elements of a matrix
def nv_el(m):
  """ get the non vanishing elments of a matrix"""
  from scipy.sparse import csc_matrix as csc
  mc = csc(m) # to coo_matrixi
  mc.eliminate_zeros()
  mc = mc.tocoo()
  data = mc.data # get data
  col = mc.col # get column index
  row = mc.row # get row index
  nv = []
  nt=len(data)
  for i in range(nt):
   # save the nonvanishing values
   nv.append([row[i]+1,col[i]+1,data[i].real,data[i].imag])
  return nv


def write_sparse(f,g):
  """ Write a sparse matrix in a file"""
  fw = open(f,"w") # open the file
  fw.write("# dimension = "+str(g.shape[0])+"\n")
  nv=nv_el(g)
  for iv in range(len(nv)):
    fw.write(str(int(nv[iv][0]))+'   ')
    fw.write(str(int(nv[iv][1]))+'   ')
    fw.write('{0:.8f}'.format(float(nv[iv][2]))+'   ')
    fw.write('{0:.8f}'.format(float(nv[iv][3]))+'   ')
    fw.write('  !!!  i  j   Real   Imag\n')
  fw.close()




def read_sparse(f,sparse=True):
  """Read green function from a file"""
  l = open(f,"r").readlines()[0] # first line
  d = int(l.split("=")[1])
  m = np.genfromtxt(f)
  if not sparse:
# create matrix  
    g = np.matrix([[0.0j for i in range(d)] for j in range(d)]) 
    for r in m:
      i = int(r[0])-1
      j = int(r[1])-1
      ar = r[2]
      ai = r[3]
      g[i,j] = ar +1j*ai # store element
  if sparse:
    from scipy.sparse import coo_matrix
    g = coo_matrix([[0.0j for i in range(d)] for j in range(d)]) 
    row = np.array([0 for i in range(len(m))])
    col = np.array([0 for i in range(len(m))])
    data = np.array([0j for i in range(len(m))])
    for i in range(len(m)):
      r = m[i]
      row[i] = int(r[0])-1
      col[i] = int(r[1])-1
      ar = r[2]
      ai = r[3]
      data[i] = ar +1j*ai # store element
    g.col = col
    g.row = row
    g.data = data
  return g # return green function





def gauss_inverse(m,i=0,j=0,test=False):
    """ Calculates the inverso of a block diagonal
        matrix """
    return block_inverse(m,i=i,j=j)
#  try: from .gauss_invf90 import gauss_inv as ginv
#  except: 
#  test = True # Ups, this might blow up
#  if test: # check whether the inversion worked 
#    return block_inverse(m,i=i,j=j)
#  nb = len(m) # number of blocks
#  ca = [None for ii in range(nb)]
#  ua = [None for ii in range(nb-1)]
#  da = [None for ii in range(nb-1)]
#  for ii in range(nb): # diagonal part
#    ca[ii] = m[ii][ii]
#  for ii in range(nb-1):
#    ua[ii] = m[ii][ii+1]
#    da[ii] = m[ii+1][ii]
#  # in case you use the -1 notation of python
#  if i<0: i += nb 
#  if j<0: j += nb 
#  # now call the actual fortran routine
#  mout = ginv(ca,da,ua,i+1,j+1)
#  mout = np.matrix(mout)
#  return mout




def block_inverse(m,i=0,j=0):
    """ Calculate a certain element of the inverse of a block matrix"""
    from scipy.sparse import csc_matrix,bmat
    nb = len(m) # number of blocks
    if i<0: i += nb 
    if j<0: j += nb 
    mt = [[None for ii in range(nb)] for jj in range(nb)]
    for ii in range(nb): # diagonal part
      mt[ii][ii] = csc_matrix(m[ii][ii])
    for ii in range(nb-1):
      mt[ii][ii+1] = csc_matrix(m[ii][ii+1])
      mt[ii+1][ii] = csc_matrix(m[ii+1][ii])
    mt = bmat(mt).todense() # create dense matrix
    # select which elements you need
    ilist = [m[ii][ii].shape[0] for ii in range(i)] 
    jlist = [m[jj][jj].shape[1] for jj in range(j)] 
    imin = int(np.sum(ilist))
    jmin = int(np.sum(jlist))
    mt = algebra.inv(mt) # calculate inverse
    imax = imin + m[i][i].shape[0]
    jmax = jmin + m[j][j].shape[1]
    mo = [ [mt[ii,jj] for jj in range(jmin,jmax)] for ii in range(imin,imax) ] 
    mo = np.matrix(mo)
    return mo



def full_inverse(m):
    """ Calculate a certain element of the inverse of a block matrix"""
    from scipy.sparse import csc_matrix,bmat
    nb = len(m) # number of blocks
    if i<0: i += nb
    if j<0: j += nb
    mt = [[None for ii in range(nb)] for jj in range(nb)]
    for ii in range(nb): # diagonal part
      mt[ii][ii] = csc_matrix(m[ii][ii])
    for ii in range(nb-1):
      mt[ii][ii+1] = csc_matrix(m[ii][ii+1])
      mt[ii+1][ii] = csc_matrix(m[ii+1][ii])
    mt = bmat(mt).todense() # create dense matrix
    return algebra.inv(mt) # calculate inverse





def green_surface_cells(gs,hop,ons,delta=1e-2,e=0.0,n=0):
    """Compute the surface Green's function for several unit cells"""
    hopH = algebra.H(hop) # Hermitian
    ez = (e+1j*delta)*np.identity(ons.shape[0]) # energy
    gt = np.zeros(ons.shape[0],dtype=np.complex) # energy
    sigmar = hop@gs@algebra.H(hop) # of the infinite right part
    out = []
    for i in range(n):
      sigmal = algebra.H(hop)@gt@hop # selfenergy
      # couple infinite right to finite left
      gemb = algebra.inv(ez - ons - sigmal- sigmar) # full dyson equation
      # compute surface spectral function of the left block only
      gt = algebra.inv(ez - ons - sigmal) # return Dyson equation
      out.append(gemb) # store this green's function
    return out # return green's functions





def interface(h1,h2,k=[0.0,0.,0.],energy=0.0,delta=0.01):
  """Get the Green function of an interface"""
  from scipy.sparse import csc_matrix as csc
  from scipy.sparse import bmat
  gs1,sf1 = green_kchain(h1,k=k,energy=energy,delta=delta,
                   only_bulk=False,reverse=True) # surface green function 
  gs2,sf2 = green_kchain(h2,k=k,energy=energy,delta=delta,
                   only_bulk=False,reverse=False) # surface green function 
  #############
  ## 1  C  2 ##
  #############
  # Now apply the Dyson equation
  (ons1,hop1) = get1dhamiltonian(h1,k,reverse=True) # get 1D Hamiltonian
  (ons2,hop2) = get1dhamiltonian(h2,k,reverse=False) # get 1D Hamiltonian
  havg = (algebra.dagger(hop1) + hop2)/2. # average hopping
  ons = bmat([[csc(ons1),csc(havg)],[csc(havg.H),csc(ons2)]]) # onsite
  self2 = bmat([[csc(ons1)*0.0,None],[None,csc(hop2@sf2@hop2.H)]])
  self1 = bmat([[csc(hop1@sf1@hop1.H),None],[None,csc(ons2)*0.0]])
  # Dyson equation
  ez = (energy+1j*delta)*np.identity(ons1.shape[0]+ons2.shape[0]) # energy
  ginter = (ez - ons - self1 - self2).I # Green function
  # now return everything, first, second and hybrid
  return (gs1,sf1,gs2,sf2,ginter)


def interface_multienergy(h1,h2,k=[0.0,0.,0.],energies=[0.0],delta=0.01,
        dh1=None,dh2=None):
  """Get the Green function of an interface"""
  from scipy.sparse import csc_matrix as csc
  from scipy.sparse import bmat
  fun1 = green_kchain_evaluator(h1,k=k,delta=delta,hs=None,
                   only_bulk=False,reverse=True) # surface green function 
  fun2 = green_kchain_evaluator(h2,k=k,delta=delta,hs=None,
                   only_bulk=False,reverse=False) # surface green function 
  out = [] # output
  for energy in energies: # loop
    gs1,sf1 = fun1(energy)
    gs2,sf2 = fun2(energy)
    #############
    ## 1  C  2 ##
    #############
    # Now apply the Dyson equation
    (ons1,hop1) = get1dhamiltonian(h1,k,reverse=True) # get 1D Hamiltonian
    (ons2,hop2) = get1dhamiltonian(h2,k,reverse=False) # get 1D Hamiltonian
    havg = (algebra.dagger(hop1) + hop2)/2. # average hopping
    if dh1 is not None: ons1 = ons1 + dh1
    if dh2 is not None: ons2 = ons2 + dh2
    ons = bmat([[csc(ons1),csc(havg)],[csc(dagger(havg)),csc(ons2)]]) # onsite
    self2 = bmat([[csc(ons1)*0.0,None],[None,csc(hop2@sf2@dagger(hop2))]])
    self1 = bmat([[csc(hop1@sf1@dagger(hop1)),None],[None,csc(ons2)*0.0]])
    # Dyson equation
    ez = (energy+1j*delta)*np.identity(ons1.shape[0]+ons2.shape[0]) # energy
    ginter = algebra.inv(ez - ons - self1 - self2) # Green function
    # now return everything, first, second and hybrid
    out.append([gs1,sf1,gs2,sf2,ginter])
  return out # return output





def surface_multienergy(h1,k=[0.0,0.,0.],energies=[0.0],reverse=True,**kwargs):
  """Get the Green function of an interface"""
  from scipy.sparse import csc_matrix as csc
  from scipy.sparse import bmat
  fun1 = green_kchain_evaluator(h1,k=k,
                   only_bulk=False,reverse=reverse,
                   **kwargs) # surface green function 
  out = [] # output
  from . import parallel
  def fp(x):
      gs1,sf1 = fun1(x)
      return [sf1,gs1]
  out = parallel.pcall(fp,energies)
#  for energy in energies: # loop
#    gs1,sf1 = fun1(energy)
#    out.append([sf1,gs1])
  return out # return output















def supercell_selfenergy(h,e=0.0,delta=1e-3,nk=100,nsuper=[1,1],
                             gtype="bulk"):
  """Calculates the selfenergy of a certain supercell """
  h.turn_dense() # dense mode
  if nsuper==1: # a single unit cell 
      return bloch_selfenergy(h,energy=e,delta=delta,nk=nk,
              mode="renormalization",gtype=gtype)
  if gtype!="bulk": return NotImplemented # not implemented
  if h.dimensionality>2: return NotImplemented
  try:   # if two number given
    nsuper1 = nsuper[0]
    nsuper2 = nsuper[1]
  except: # if only one number given
    nsuper1 = nsuper
    nsuper2 = nsuper
#  print("Supercell",nsuper1,"x",nsuper2)
  ez = e + 1j*delta # create complex energy
  from . import dyson
  g = dyson.dyson(h,[nsuper1,nsuper2],nk,ez)
  g = np.matrix(g) # convert to matrix
  # create hamiltonian of the supercell
  from .embedding import onsite_supercell
  intrasuper = onsite_supercell(h,nsuper)
  eop = np.matrix(np.identity(g.shape[0],dtype=np.complex))*(ez)
  selfe = eop - intrasuper - algebra.inv(g)
  return g,selfe







def green_generator(h,nk=20):
  """Returns a function capable of calculating the Green function
  at a certain energy, by explicity summing the k-dependent Green functions"""
  if h.dimensionality != 2: raise # only for 2d
  shape = h.intra.shape # shape
  hkgen = h.get_hk_gen() # get the Hamiltonian generator
  wfs = np.zeros((nk*nk,shape[0],shape[0]),dtype=np.complex) # allocate vector
  es = np.zeros((nk*nk,shape[0])) # allocate vector, energies
  ks = np.zeros((nk*nk,2)) # allocate vector, energies
  ii = 0 # counter
  for ik in np.linspace(0.,1.,nk,endpoint=False): # loop
    for jk in np.linspace(0.,1.,nk,endpoint=False): # loop
      estmp,wfstmp = algebra.eigh(hkgen([ik,jk])) # get eigens
#      estmp,wfstmp = lg.eigh(hkgen(np.random.random(2))) # get eigens
      es[ii,:] = estmp.copy() # copy
      ks[ii,:] = np.array([ik,jk]) # store
      wfs[ii,:,:] = wfstmp.transpose().copy() # copy
      ii += 1 # increase counter
  # All the wavefunctions have been calculate
  # Now create the output function
  from scipy.integrate import simps
  def getgreen(energy,delta=0.001):
    """Return the Green function"""
    zero = np.array(np.zeros(shape,dtype=np.complex)) # zero matrix
    zero = getgreen_jit(wfs,es,energy,delta,zero)
    ediag = np.array(np.identity(shape[0]))*(energy + delta*1j)
    selfenergy = ediag - h.intra - algebra.inv(zero)
    return zero,selfenergy
  return getgreen # return function

@jit(nopython=True)
def getgreen_jit(wfs,es,energy,delta,zero):
    """Jit summation of Bloch Green's function"""
    shape = wfs[0].shape
    for ii in range(len(es)): # loop over kpoints
      v = energy + delta*1j - es[ii,:] # array
      C = zero*0.0 # initilaize
      for j in range(len(v)): C[j,j] = 1./v[j]
      A = wfs[ii,:,:] # get the matrix with wavefunctions
      zero += np.conjugate(A).T@C@A # add contribution
    zero /= len(es) # normalize
    return zero




def green_operator(h0,operator=None,e=0.0,delta=1e-3,nk=10,
        gmode="adaptive"):
    """Return the integration of an operator times the Green function"""
    h = h0.copy()
    h.turn_dense()
    hkgen = h.get_hk_gen() # get generator
    iden = np.identity(h.intra.shape[0],dtype=np.complex)
    from . import klist
    ks = klist.kmesh(h.dimensionality,nk=nk) # klist
    out = 0.0 # output
    if callable(operator): # callable operator
      for k in ks: # loop over kpoints
        hk = hkgen(k) # Hamiltonian
        o0 = algebra.inv(iden*(e+1j*delta) - hk) # Green's function
        if callable(operator): o1 = operator(k)
        else: o1 = operator
        out += -np.trace(o0@o1).imag # Add contribution
      out /= len(ks) # normalize
    else:
      g = bloch_selfenergy(h,energy=e,delta=delta,mode=gmode)[0] 
      if operator is None: out = -np.trace(np.array(g)).imag
      else: out = -np.trace(np.array(g)@operator).imag
    return out



def GtimesO(g,o,k=[0.,0.,0.]):
    """Green function times operator"""
    o = algebra.todense(o) # convert to dense operator if possible
    if o is None: return g # return Green function
    elif type(o)==type(g): return g@o # return
    elif callable(o): return o(g,k=k) # call the operator
    else:
        print(type(g),type(o))
        raise



