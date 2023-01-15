import numpy as np
from scipy.sparse import csc_matrix,bmat,coo_matrix
from . import parallel


def collect_hopping(h):
    """Collect the hoppings in a new list"""
    td = dict() # dictionary
    zero = h.intra*0.0 # initialize
    for t in h.hopping: td[tuple(t.dir)] = zero.copy() # initialize
    for t in h.hopping: 
        td[tuple(t.dir)] += t.m # add
    ts = [] # empty list
    for key in td:
        if np.max(np.abs(td[key]))>1e-6:
          ts.append(Hopping(d=np.array(key),m=td[key]))

    return ts




class Hopping(): 
  def __init__(self,d=None,m=None):
    self.dir = np.array(d)
    self.m = m


def turn_multicell(h):
    """Transform a normal hamiltonian into a multicell hamiltonian"""
    if h.is_multicell: return h # if it is already multicell
    ho = h.copy() # copy hamiltonian
    # directions
    dirs = []
    if h.dimensionality == 0: ts = []
    elif h.dimensionality == 1: # one dimensional
      dirs.append(np.array([1,0,0]))
      ts = [h.inter.copy()]
      ho.inter = None
    elif h.dimensionality == 2: # two dimensional
      dirs.append(np.array([1,0,0]))
      dirs.append(np.array([0,1,0]))
      dirs.append(np.array([1,1,0]))
      dirs.append(np.array([1,-1,0]))
      ts = [h.tx.copy(),h.ty.copy(),h.txy.copy(),h.txmy.copy()]
      del ho.tx
      del ho.ty
      del ho.txy
      del ho.txmy
    else: raise
    dd = dict() # dictionary
    dd[(0,0,0)] = h.intra
    for (d,t) in zip(dirs,ts): 
        dd[tuple(d)] = t
        dd[tuple(-np.array(d))] = np.conjugate(t).T
    return set_dictionary(ho,dd) # return this Hamiltonian


def set_multihopping(ho,dd):
    """Set a multihopping as the Hamiltonian"""
    dd = dd.get_dict()
    return set_dictionary(ho,dd) # return this Hamiltonian



def set_dictionary(ho,dd):
    """Set a dictionary as the Hamiltonian"""
    ho.intra = dd[(0,0,0)].copy() # intracell
    hoppings = [] # list of hoppings
    for d in dd: # loop over hoppings
        t = dd[d] # matrix
        if d==(0,0,0): continue
        hopping = Hopping() # create object
        hopping.m = t.copy()
        hopping.dir = np.array(d)
        hoppings.append(hopping) # store
    ho.hopping = hoppings # store all the hoppings
    ho.is_multicell = True # multicell hamiltonian
    return ho




def generate_get_tij(h):
    """Get the hopping between cells with certain indexes"""
    hdict = h.get_multihopping().get_dict() # generate the multihopping object
    mzero = hdict[(0,0,0)]*0.
    def fun(rij=np.array([0.,0.,0.]),zero=False):
        drij = tuple([int(round(ir)) for ir in rij]) # put as integer
        if drij in hdict: return hdict[tuple(drij)]
        else:
            if zero: return mzero
            else: return None
    return fun

def hk_gen(h):
  """Generate a k dependent hamiltonian"""
  if not h.is_multicell:
      h = h.get_multicell()
  # get the non zero hoppings
  hopping = [] # empty list
  for t in h.hopping: # loop
    if h.is_sparse:
      if np.sum(np.abs(coo_matrix(t.m).data))>1e-7: hopping.append(t) # store this hopping
    else:
      if np.sum(np.abs(t.m))>1e-7: hopping.append(t) # store this hopping
  if h.dimensionality == 0: return lambda k: h.intra
  elif h.dimensionality == 1: # one dimensional
    def hk(k):
      """k dependent hamiltonian, k goes from 0 to 1"""
      mout = h.intra.copy() # intracell term
      for t in hopping: # loop over matrices
        tk = t.m * h.geometry.bloch_phase(t.dir,k) # k hopping
        mout = mout + tk 
      return mout
    return hk  # return the function
  elif h.dimensionality == 2: # two dimensional
    def hk(k):
      """k dependent hamiltonian, k goes from 0 to 1"""
      mout = h.intra.copy() # intracell term
      for t in hopping: # loop over matrices
        tk = t.m * h.geometry.bloch_phase(t.dir,k) # k hopping
        mout = mout + tk 
      return mout
    return hk  # return the function
  elif h.dimensionality == 3: # three dimensional
    def hk(k):
      """k dependent hamiltonian, k goes from 0 to 1"""
      mout = h.intra.copy() # intracell term
      for t in h.hopping: # loop over matrices
        tk = t.m * h.geometry.bloch_phase(t.dir,k) # k hopping
        mout = mout + tk 
      return mout
    return hk  # return the function
  else: raise







def turn_spinful(h,enforce_tr=False):
  """Turn a hamiltonian spinful"""
  from .increase_hilbert import spinful
  from .superconductivity import time_reversal
  if h.has_eh: raise
  if h.has_spin: return # return if already has spin
  h.has_spin = True # put spin
  def fun(m):
    return spinful(m)
  h.intra = fun(h.intra) # spinful intra
  for i in range(len(h.hopping)): 
    h.hopping[i].m = fun(h.hopping[i].m) # spinful hopping




def bulk2ribbon(hin,n=10,sparse=True,nxt=6,ncut=6):
  """ Create a ribbon hamiltonian object"""
  if not hin.is_multicell: h = turn_multicell(hin)
  else: h = hin # nothing othrwise
  hr = h.copy() # copy hamiltonian
  if sparse: hr.is_sparse = True # sparse output
  hr.dimensionality = 1 # reduce dimensionality
  # stuff about geometry
  hr.geometry = h.geometry.supercell((1,n)) # create supercell
  hr.geometry.dimensionality = 1
  hr.geometry.a1 = h.geometry.a1 # add the unit cell vector
  from . import sculpt # rotate the geometry
  hr.geometry = sculpt.rotate_a2b(hr.geometry,hr.geometry.a1,np.array([1.,0.,0.]))
  hr.geometry.celldis = hr.geometry.a1[0]
  get_tij = generate_get_tij(h) # return a function to abtain the hoppings
  def superhopping(dr=[0,0,0]): 
    """ Return a matrix with the hopping of the supercell"""
    intra = [[None for i in range(n)] for j in range(n)] # intracell term
    for ii in range(n): # loop over ii
      for jj in range(n): # loop over jj
        d = np.array([dr[0],ii-jj+dr[1],dr[2]])
        if d.dot(d)>ncut*ncut: continue # skip iteration
        m = get_tij(rij=d) # get the matrix
        if m is not None: intra[ii][jj] = csc_matrix(m) # store
        else: 
          if ii==jj: intra[ii][jj] = csc_matrix(h.intra*0.)
    intra = bmat(intra) # convert to matrix
    if not sparse: intra = intra.todense() # dense matrix
    return intra
  # get the intra matrix
  hr.intra = superhopping()
  # now do the same for the interterm
  hoppings = [] # list of hopings
  for i in range(-nxt,nxt+1): # loop over hoppings
    if i==0: continue # skip the 0
    d = np.array([i,0.,0.])
    hopp = Hopping() # create object
    hopp.m = superhopping(dr=d) # get hopping of the supercell
    hopp.dir = d
    hoppings.append(hopp)
  hr.hopping = hoppings # store the list
  hr.dimensionality = 1
  return hr 




def rotate90(h):
  """ Rotate 90 degrees the Hamiltonian"""
  ho = turn_multicell(h) # copy Hamiltonian
  hoppings = []
  for i in range(len(ho.hopping)):
    tdir = ho.hopping[i].dir 
    ho.hopping[i].dir = np.array([tdir[1],tdir[0],tdir[2]]) # new direction
  ho.geometry.a1,ho.geometry.a2 = h.geometry.a2,h.geometry.a1
  return ho


def rotate(h,m):
  """ Rotate the Hamiltonian"""
  ho = h.copy() # duplicate Hamiltonian
  m = np.matrix(m) # convert to matrix
  for i in range(len(h.hopping)):
    tdir = np.array(h.hopping[i].dir)*m.T # rotation of the direction
    tdir = [tdir[0,0],tdir[0,1],tdir[0,2]]
#    print(h.hopping[i].dir,tdir)
    ho.hopping[i].dir = tdir # new direction
  ho.hopping_dict = dict() # new dictionary
  ho.geometry.a1,ho.geometry.a2 = h.geometry.a2*m.T,ho.geometry.a1*m.T
  return ho


def basis_change(h,R):
  """Perform a change of basis in the Hamiltonian"""
  ho = h.copy() # duplicate Hamiltonian
  R = np.matrix(R) # convert to matrix
  for i in range(len(h.hopping)):
    m = h.hopping[i].m # rotation of the direction
    ho.hopping[i].m = R.H*m*R # Hamiltonian in new basis
  return ho





def clean(h,cutoff=0.0001):
  """Remove hoppings smaller than a certain quantity"""
  ho = h.copy() # copy hamiltonian
  raise 



def supercell_hamiltonian(hin,nsuper=[1,1,1],sparse=True,ncut=3,
                             **kwargs):
  """ Create a ribbon hamiltonian object"""
#  raise # there is something wrong with this function
#  print("This function might have something wrong")
  if not hin.is_multicell: h = turn_multicell(hin)
  else: h = hin # nothing otherwise
  hr = h.copy() # copy hamiltonian
  if sparse: hr.is_sparse = True # sparse output
  # stuff about geometry
  hr.geometry = h.geometry.get_supercell(nsuper,**kwargs) # create supercell
  n = nsuper[0]*nsuper[1]*nsuper[2] # number of cells in the supercell
  pos = [] # positions inside the supercell
  for i in range(nsuper[0]):
    for j in range(nsuper[1]):
      for k in range(nsuper[2]):
        pos.append(np.array([i,j,k])) # store position inside the supercell
  zero = csc_matrix(np.zeros(h.intra.shape,dtype=np.complex)) # zero matrix
  get_tij = generate_get_tij(h) # return a function to abtain the hoppings
  def superhopping(dr=[0,0,0]): 
    """ Return a matrix with the hopping of the supercell"""
    rs = [dr[0]*nsuper[0],dr[1]*nsuper[1],dr[2]*nsuper[2]] # supercell vector
    intra = [[None for i in range(n)] for j in range(n)] # intracell term
    for ii in range(n): intra[ii][ii] = zero.copy() # zero

    for ii in range(n): # loop over cells
      for jj in range(n): # loop over cells
        d = pos[jj] + np.array(rs) -pos[ii] # distance
      #  if d.dot(d)>ncut*ncut: continue # skip iteration
        m = get_tij(rij=d) # get the matrix
        if m is not None: 
          intra[ii][jj] = csc_matrix(m) # store
    intra = csc_matrix(bmat(intra)) # convert to matrix
    if not sparse: intra = intra.todense() # dense matrix
    return intra
  # get the intra matrix
  hr.intra = superhopping()
  # now do the same for the interterm
  hoppings = [] # list of hopings
  for i in range(-ncut,ncut+1): # loop over hoppings
    for j in range(-ncut,ncut+1): # loop over hoppings
      for k in range(-ncut,ncut+1): # loop over hoppings
        if i==j==k==0: continue # skip the intraterm
        dr = np.array([i,j,k]) # set as array
        hopp = Hopping() # create object
        hopp.m = superhopping(dr=dr) # get hopping of the supercell
        hopp.dir = dr
        if np.sum(np.abs(hopp.m))>0.00000001: # skip this matrix
          hoppings.append(hopp)
        else: pass
      hr.hopping = hoppings # store the list
  return hr 



from .current import derivative



def parametric_hopping_hamiltonian(h,cutoff=5,fc=None,rcut=5.0):
  """ Gets a first neighbor hamiltonian"""
  from .neighbor import parametric_hopping
  if fc is None:
    rcut = 2.1 # stop in this neighbor
    def fc(r1,r2):
      r = r1-r2
      r = r.dot(r)
      if 0.9<r<1.1: return 1.0
      else: return 0.0
  r = h.geometry.r    # x coordinate 
  g = h.geometry
  h.is_multicell = True 
# first neighbors hopping, all the matrices
  a1, a2, a3 = g.a1, g.a2, g.a3
  h.intra = h.spinless2full(parametric_hopping(r,r,fc)) # intra matrix
  # generate directions
  dirs = h.geometry.neighbor_directions(n=cutoff) # directions of the hoppings
  # generate hoppings
  h.hopping = [] # empty list
  for d in dirs: # loop over directions
        i1,i2,i3 = d[0],d[1],d[2] # extract indexes
        if i1==0 and i2==0 and i3==0: continue
        t = Hopping() # hopping class
        da = a1*i1+a2*i2+a3*i3 # direction
        r2 = [ri + da for ri in r]
        if not close_enough(r,r2,rcut=rcut): # check if we can skip this one
#          print("Skipping hopping",[i1,i2,i3])
          continue
        t.m = h.spinless2full(parametric_hopping(r,r2,fc))
        t.dir = [i1,i2,i3] # store direction
        if np.sum(np.abs(t.m))>0.00001: h.hopping.append(t) # append 
  return h







def parametric_matrix(h,cutoff=2,fm=None):
  """ Gets a first neighbor hamiltonian"""
  from .neighbor import parametric_hopping
  if fm is None: raise
  r = h.geometry.r    # x coordinate 
  g = h.geometry
  h.is_multicell = True
# first neighbors hopping, all the matrices
  a1, a2, a3 = g.a1, g.a2, g.a3
  h.intra = h.spinless2full(fm(r,r)) # intra matrix
  # generate directions
  dirs = h.geometry.neighbor_directions(n=cutoff) # directions of the hoppings
  # generate hoppings
  h.hopping = [] # empty list
  def gett(d):
        i1,i2,i3 = d[0],d[1],d[2] # extract indexes
        if i1==0 and i2==0 and i3==0: return None
        t = Hopping() # hopping class
        da = a1*i1+a2*i2+a3*i3 # direction
        r2 = [ri + da for ri in r]
        t.m = h.spinless2full(fm(r,r2))
        t.dir = [i1,i2,i3] # store direction
        if np.sum(np.abs(t.m))>0.00001: return t
        else: return None
#  from . import parallel
  ts = parallel.pcall(gett,dirs) # get hoppings
#  ts = [gett(d) for d in dirs] # get hoppings in serie
  ts = [x for x in ts if x is not None] # remove Nones
  h.hopping = ts # store
  return h



# rename this function
first_neighbors = parametric_hopping_hamiltonian

def read_from_file(input_file="hamiltonian.wan"):
  """Generate a function that return hopping matrices"""
  mt = np.genfromtxt(input_file) # get file
  m = mt.transpose() # transpose matrix
  nmax = int(np.max([np.max(m[i])for i in range(3)]))
  ncells = [nmax,nmax,nmax] # number of cells
  # read the hamiltonian matrices
  tlist = []
  norb = np.max([np.max(np.abs(m[3])),np.max(np.abs(m[4]))])
  norb = int(norb)
  zeros = np.matrix(np.zeros((norb,norb),dtype=np.complex)) # zero matrix
  def get_t(i,j,k):
    mo = zeros.copy() # copy matrix
    found = False
    for l in mt: # look into the file
      if i==int(l[0]) and j==int(l[1]) and k==int(l[2]): # right hopping
        mo[int(l[3])-1,int(l[4])-1] = l[5] + 1j*l[6] # store element
        found  = True # matrix has been found
    if found:  return mo # return the matrix
    else: return None # return nothing if not found
  tdict = dict() # dictionary
  for i in range(-ncells[0],ncells[0]+1):
    for j in range(-ncells[1],ncells[1]+1):
      for k in range(-ncells[2],ncells[2]+1):
        matrix = get_t(i,j,k) # read the matrix
        if matrix is None: continue # skip if matrix not found
        tdict[(i,j,k)] = matrix # store matrix
  # now create the function
  def get_hopping(i,j,k):
    """Get hopping in a certain direction"""
    try: # look into the dictionary
      return tdict[(i,j,k)]
    except: return zeros  # return zero matrix if not found
  # check that the function returns a Hermitian Hamiltonian
  for i in range(-ncells[0],ncells[0]+1):
    for j in range(-ncells[1],ncells[1]+1):
      for k in range(-ncells[2],ncells[2]+1):
        dm = get_hopping(i,j,k) - get_hopping(-i,-j,-k).H
        if np.sum(np.abs(dm))>0.0001: raise
  print("Hopping generator is Hermitian")
  return get_hopping # return the function
     

def save_multicell(ns,ms,output_file="multicell.sym"):
  """Save a multicell Hamiltonian in a file"""
  fo = open(output_file,"w") # open file
  for (n,m) in zip(ns,ms): # loop over hoppings
    for i in range(m.shape[0]):
      for j in range(m.shape[0]):
        fo.write(str(n[0])+"    ") # cell number
        fo.write(str(n[1])+"    ") # cell number
        fo.write(str(n[2])+"    ") # cell number
        fo.write(str(i+1)+"    ") # index
        fo.write(str(j+1)+"    ") # index
        fo.write(str(m[i,j].real)+"    ") # index
        fo.write(str(m[i,j].imag)+"\n") # index
  fo.close() # close file



def pairs2hopping(ps):
  """Takes as input a list of pairs vector/matrix, return a list of classess"""
  hopping = [Hopping(d,m) for (d,m) in ps] # empty list
  return hopping



def close_enough(rs1,rs2,rcut=2.0):
  """CHeck if two sets of positions are at a distance
  at least rcut"""
  rcut2 = rcut*rcut # square of the distance
  for ri in rs1:
    for rj in rs2:
      dr = ri - rj # vector
      dr = dr.dot(dr) # distance
      if dr<rcut2: return True
  return False



def turn_no_multicell(h):
  """Converts a Hamiltonian into the non multicell form"""
  if not h.is_multicell: return h # Hamiltonian is already fine
  ho = h.copy() # copy Hamiltonian
  ho.is_multicell = False
  if ho.dimensionality==0: pass
  elif ho.dimensionality==1:
    ho.inter = ho.intra*0.
  elif ho.dimensionality==2:
    ho.tx = ho.intra*0.
    ho.txy = ho.intra*0.
    ho.txmy = ho.intra*0.
    ho.ty = ho.intra*0.
  else: raise
  for t in h.hopping: # loop over hoppings
    if h.dimensionality==0: pass # one dimensional
    elif h.dimensionality==1: # one dimensional
      if t.dir[0]==1 and t.dir[1]==0 and t.dir[2]==0: # 
        ho.inter = t.m # store
      elif np.sum(np.abs(t.m))>0.0001 and np.max(np.abs(t.dir))>1: raise # Uppps, not possible
    elif h.dimensionality==2: # two dimensional
      if t.dir[0]==1 and t.dir[1]==0 and t.dir[2]==0: # 
        ho.tx = t.m # store
      elif t.dir[0]==0 and t.dir[1]==1 and t.dir[2]==0: # 
        ho.ty = t.m # store
      elif t.dir[0]==1 and t.dir[1]==1 and t.dir[2]==0: # 
        ho.txy = t.m # store
      elif t.dir[0]==1 and t.dir[1]==-1 and t.dir[2]==0: # 
        ho.txmy = t.m # store
      elif np.sum(np.abs(t.m))>0.0001 and np.max(np.abs(t.dir))>1: raise # Uppps, not possible
    else: raise
  ho.hopping = [] # empty list
  return ho



def kchain(h,k=[0.,0.,0.]):
  """Return the onsite and hopping for a particular k"""
  if not h.is_multicell: h = h.get_multicell()
  dim = h.dimensionality # dimensionality
  if dim==1: # 1D
      for t in h.hopping:
          if t.dir[0]==1: return h.intra,t.m
      raise
  elif dim>1: # 2D or 3D
    intra = np.zeros(h.intra.shape) # zero amtrix
    inter = np.zeros(h.intra.shape) # zero amtrix
    intra = h.intra # initialize
    for t in h.hopping: # loop over hoppings
      tk = t.m * h.geometry.bloch_phase(t.dir,k) # k hopping
      if t.dir[dim-1]==0: intra = intra + tk # add contribution 
      if t.dir[dim-1]==1: inter = inter + tk # add contribution 
    return intra,inter 
  else: raise




def get_hopping_dict(h):
    """Return the hopping dictionary"""
    h = h.get_multicell()
    out = dict()
    out[(0,0,0)] = h.intra.copy()
    for t in h.hopping: 
        if np.max(np.abs(t.m))>1e-7:
          out[tuple(t.dir)] = t.m.copy() # store
    return out # return


#
#def multiply_hopping_dict(hop1,hop2):
#    """Multiply two hopping dictionaries"""
#    out = dict() # create dictionary
#    for key1 in hop1:
#        for key2 in hop2:
#            key = tuple(np.array(key1) + np.array(key2))
#            m = hop1[key1]@hop2[key2] # multiply
#            if np.max(np.abs(m))>1e-6: # if non-zero
#                if key in out: out[key] = out[key] + m # add
#                else: out[key] = m # store
#    return out # return output
#
#
#
#def add_hopping_dict(hop1,hop2):
#    """Multiply two hopping dictionaries"""
#    out = dict() # create dictionary
#    keys = [key1 for key1 in hop1]
#    for key2 in hop2:
#        if key2 not in keys: keys.append(key2) # store
#    for key in keys:
#        m = 0 # initialize
#        if key in hop1: m = m + hop1[key]
#        if key in hop2: m = m + hop2[key]
#        out[key] = m # store
#    return out
#
#
#
#class MultiHopping():
#    """Class for a multihopping"""
#    def __init__(self,a=dict()):
#        if type(a)==dict: # dictionary type
#            self.dict = a # dictionary
#        else: return NotImplemented
#    def __add__(self,a):
#        if type(a)!=MultiHopping: return NotImplemented
#        out = MultiHopping() # create a new object
#        out.dict = add_hopping_dict(self.dict,a.dict)
#        return out
#    def __mul__(self,a):
#        from .hamiltonians import is_number
#        if type(a)==MultiHopping: 
#            out = MultiHopping() # create a new object
#            out.dict = multiply_hopping_dict(self.dict,a.dict)
#            return out
#        elif is_number(a):
#            out = MultiHopping() # create a new object
#            for key in self.dict:
#                out.dict[key] = a*self.dict[key] 
#            return out
#        else: return NotImplemented
#    def __rmul__(self,a): return self*a
#    def __neg__(self): return (-1)*self
#    def __sub__(self,a): return self + (-self)
#    def get_dict(self):
#        return self.dict # dictionary
#    def get_dagger(self):
#        out = MultiHopping() # create a new object
#        for key in self.dict:
#            out.dict[key] = np.conjugate(self.dict[key]).T
#        return out
#
#
#
#
#

from .multihopping import MultiHopping

