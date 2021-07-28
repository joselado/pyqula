from __future__ import print_function
import numpy as np
import scipy.linalg as lg
from . import wannier
from . import multicell
from . import angular
from scipy.sparse import csc_matrix, bmat

debug = True

if debug:
  fdebug = open(".debug.summetrize","w")

# routines to get a symmetric Hamiltonian from Wannier

def read_orbitals(input_file="orbitals.wan"):
  """Get the indexes and orbitals"""
  lines = open(input_file,"r").readlines()
  inds = [] # index
  atoms = [] # name
  orbs = [] # orbital
  rs = [] # coordinates in unit cell vectors
  for l in lines: # loop over lines
    l = l.split() # separte
    if "#" in l[0]: continue # comment, next iteration
    i = int(l[0])-1 # index
    ss = l[1] # atom and orbital
    ss = ss.split(",") # separate by the comma
    iatom = ss[0] # name of the atom
    iorb = ss[1] # name of the orbital
    inds.append(i) # store
    atoms.append(iatom) # store
    orbs.append(iorb) # store
    r = np.array([float(l[5]),float(l[6]),float(l[7])])
    rs.append(r) # store position
  return (inds,atoms,orbs,rs) # return the three things


def index_dictionary(inds,atoms,orbs):
  """Get a dictionary with the indexes of atom and orbital"""
  idict = dict() # create dictionary
  for (i,a,o) in zip(inds,atoms,orbs): # loop over indexes
    idict[(a,o)] = i # store index for atom and orbital
  return idict # return the dictionary


def orbital_dictionary(inds,atoms,orbs):
  """Get a dictionary with the indexes of atom and orbital"""
  odict = dict() # create dictionary
  for a in atoms: odict[a] = [] # empty list, initialize
  for (a,o) in zip(atoms,orbs): # loop over orbitals
    odict[a] += [o] # add this orbital
  return odict # return the dictionary

def get_atom_names(atoms):
  """Get names of the atoms"""
  anames = []
  for a in atoms:
    if  not a in anames: anames.append(a)
  return anames # return names



def position_dictionary(atoms,rs):
  """Get names of the atoms"""
  rdict = dict() # dictionary
  for (a,r) in zip(atoms,rs):
    rdict[a] = r # store position
  return rdict # return dictionary



def read_unit_cell(input_file="geometry.wan"):
  """Read unit cell vectors"""
  m = np.genfromtxt(input_file)
  return (m[0],m[1],m[2]) # return vectors



def extract_matrix(a1,a2,odict,idict,matrix):
  """Extract the hopping between two atoms"""
  orbs1 = odict[a1] # orbitals of the first atom
  orbs2 = odict[a2] # orbitals of the second atom
  n1 = len(orbs1) # size of the basis in the first atom
  n2 = len(orbs2) # size of the basis in the second atom
  zeros = np.matrix(np.zeros((n1,n2),dtype=np.complex)) # zero matrix  
#  print(a1,a2,orbs1)
  for ii in range(len(orbs1)): # loop over orbitals in the first atom
    for jj in range(len(orbs2)): # loop over orbitals in the second atom
      i = idict[(a1,orbs1[ii])] # index of this orbital 
      j = idict[(a2,orbs2[jj])] # index of this orbital 
      zeros[ii,jj] = matrix[i,j] # store this matrix
#      print(a1,a2,i,j,orbs1[ii],orbs2[jj])
#  raise
  return zeros # return matrix





def rotation_operator(orbs,v=np.array([1.,0.,0.])):
  """Generate a rotation matrix along the z axis, depending on the
  orbitals used. Rotates vector v to vector 1,0,0"""
  n = len(orbs) # number of orbitals
  zeros = np.matrix(np.zeros((n,n),dtype=np.complex)) # zero matrix  
  (lx,ly,lz) = angular.angular_momentum(orbs) # angular momentum matrices
  v = np.array(v)
  vv = v.dot(v) # modulus
  if vv> 0.000001:
    v = v/(np.sqrt(v.dot(v))) # normalize the vector
    angle = np.arctan2(v[1],v[0]) # get the angle
  else:
    angle = 0.0 # zero angle, to return identity
#  raise
  R = np.matrix(lg.expm(-1j*lz*angle)) # get the rotation matrix
  if np.sum(np.abs(lz-lz.H))>0.0001: raise # check that lz is Hermitian
  return R # return rotation operator
  
  


def reorder_basis(anames,odict,idict):
  """Matrix that transforms the new reordered basis into the original one"""
  n = sum([len(odict[a]) for a in anames]) # length of the basis
  zeros = np.matrix(np.zeros((n,n),dtype=np.complex)) # zero matrix  
  i = 0 # initialize counter
  for a in anames: # loop over atoms
    for o in odict[a]: # loop over orbitals
      j = idict[(a,o)] # get index in the Wannier file
      zeros[i,j] = 1.
      i += 1 # increase counter
  return zeros


def local_symmetrizer(anames,ons,odict,sym_file="symmetry.wan"):
  """Return the matrices that symmetrizes the local crystal field,
  mposing a certain symmetry"""
  stypedict = dict() # dictionary for the symmetries
  sdict = dict() # dictionary for the symmetry matrices
  lines = open(sym_file).readlines() # read the lines
  for l in lines:
    l = l.split()
    stypedict[l[0]] = l[1] # save symmetry for this atom
  for (a,m) in zip(anames,ons): # now do a loop over all the different atoms
    sname = stypedict[a] # symmetry for this atom
    if sname == "C3": nr = 3# C3 symmetry
    else: raise # not implemented
    step = 2.*np.pi/nr # 2 times pi over number of rotations
    # generate vectors
    vs = [np.array([np.cos(step*i),np.sin(step*i)]) for i in range(nr)]
    # generate rotation matrices
    Rs = [rotation_operator(odict[a],v=v) for v in vs]
    # perform all the rotations in the onsite matrix
    hrs = [R.H*m*R for R in Rs]
    hr,error = average_matrices(hrs) # perform the average
    print("Broken symmetry in",a,"is",error)
    sdict[a] = hr # store rotation matrix
  return sdict # return the dictionary
  



def symmetrize_hamiltonian(orb_file="orbitals.wan",ham_file="hamiltonian.wan",
                        cell_file="geometry.wan",nlook=[4,4,1],
                        sym_file="symmetry.wan"):
  """Write in a file the new Hamiltonian, symmetrized"""
  (inds,atoms,orbs,rs) = read_orbitals(input_file=orb_file) # read the orbitals
  anames = get_atom_names(atoms) # different atom names
  # function to get hoppings to arbitrary cells
  get_hopping = multicell.read_from_file(input_file=ham_file)
  # get the dictionaries
  idict = index_dictionary(inds,atoms,orbs)
  odict = orbital_dictionary(inds,atoms,orbs)
  rdict = position_dictionary(atoms,rs)
  # matrix that will reorder the orbitals
  T = reorder_basis(anames,odict,idict)
  # read unit cell vectors
  (a1,a2,a3) = read_unit_cell()
  # generate the symmetrizer matrices
  h0 = get_hopping(0,0,0) # onsite matrix
  ons = [extract_matrix(a,a,odict,idict,h0) for a in anames] # in each atom
  sdict = local_symmetrizer(anames,ons,odict,sym_file=sym_file) # matrices
  # now look into neighbors to get matrices 
  stored_rs = [] # distances between atoms stored
  stored_atoms = [] # distances between atoms stored
  stored_matrix = [] # distances between atoms stored
  stored_vectors = [] # vectors between atoms
  for ia in anames: # loop over atoms
    for ja in anames: # loop over atoms
      ri = rdict[ia] # position of iesim atom
      ri = ri[0]*a1 + ri[1]*a2 + ri[2]*a3 # position in real coordinates
      for n1 in range(-nlook[0],nlook[0]+1):
        for n2 in range(-nlook[1],nlook[1]+1):
          for n3 in range(-nlook[2],nlook[2]+1):
            rj = rdict[ja] # position of iesim atom
            # position in real coordinates
            rj = (rj[0]+n1)*a1 + (rj[1]+n2)*a2 + (rj[2]+n3)*a3 
            dr = rj-ri # vector between this two atoms 
            drr = np.sqrt(dr.dot(dr)) # distance between atoms
            m = get_hopping(n1,n2,n3) # hopping in this direction
            tij = extract_matrix(ia,ja,odict,idict,m) # hopping between atoms
            # special case of onsite matrices
            if ia==ja and n1==n2==n3==0:
              tij = sdict[ia] # get the symmetrized onsite matrix
            Ri = rotation_operator(odict[ia],dr) # rotate basis i
            Rj = rotation_operator(odict[ja],dr) # rotate basis j
            Rtij = Ri.H*tij*Rj # hopping matrix in the new frame
            # now store the different things needed
            stored_rs.append(drr) # store the distance
            stored_vectors.append(dr) # store the vector
            stored_atoms.append((ia,ja)) # store the atom pairs
            stored_matrix.append(Rtij) # store the rotated matrix
  # get the function that generated symmetric hoppings
  hop_gen = symmetric_hopping_generator(stored_rs,stored_vectors,
                                          stored_atoms,stored_matrix)
  # now create hoppings of the symmetric Hamiltonian
  rmax = np.sqrt(a1.dot(a1))*(nlook[0]+1)/2. # maximum distance
  print("Cutoff distance is",rmax)
  ns_list = [] # list with ns
  hops_list = [] # list with ns
  for n1 in range(-nlook[0],nlook[0]+1):
    for n2 in range(-nlook[1],nlook[1]+1):
      for n3 in range(-nlook[2],nlook[2]+1):
        hop = [[None for i in range(len(anames))] for j in range(len(anames))]
        for i in range(len(anames)): # loop over atoms
          for j in range(len(anames)): # loop over atoms
            ia = anames[i] # name of the atom
            ja = anames[j] # name of the atom
            ri = rdict[ia] # position of iesim atom
            rj = rdict[ja] # position of jesim atom
            ri = ri[0]*a1 + ri[1]*a2 + ri[2]*a3 # position in real coordinates
            rj = (rj[0]+n1)*a1 + (rj[1]+n2)*a2 + (rj[2]+n3)*a3 
            dr = rj-ri # vector between this two atoms 
            drr = np.sqrt(dr.dot(dr)) # distance between atoms
            m = hop_gen(ia,ja,dr)# get the symmetric hopping
            Ri = rotation_operator(odict[ia],dr) # rotate basis i
            Rj = rotation_operator(odict[ja],dr) # rotate basis j
            if m is not None:
              m = Ri*m*Rj.H # hopping matrix in the new frame, opposite rot
              if drr>rmax: m *= 0. # reached maximum distance
              hop[i][j] = csc_matrix(m) # store the matrix 
        hop = bmat(hop).todense() # create dense matrix
        hop = T.H *hop *T # convert to the original order
        hops_list.append(hop) # store hopping
        ns_list.append([n1,n2,n3]) # store indexes
  multicell.save_multicell(ns_list,hops_list)
      

def symmetric_hopping_generator(rs,drs,atoms,matrix):
  """Returns a function that generates hoppings with the symmetry of the
  lattice, inputs are names of the atoms and distance between them"""
  ijdict = dict() # dictionary for hopping between atoms ij
  for ij in atoms: ijdict[ij] = [] # initialize as empty list
  # now do a loop over all inputs, storing the matrices
  for (r,dr,ij,m) in zip(rs,drs,atoms,matrix): # loop over inputs
    ijdict[ij].append((r,dr,m)) # store distance and matrix
  ijs = [] # empty list
  for ij in atoms: # loop over atoms
    if ij not in ijs: ijs.append(ij) # store the different pairs
  tdict = dict() # dictionary for the hoppings
  for ij in ijs: # loop over different pairs of atoms
    rm = ijdict[ij] # get the distances and matrices for this pair
    difr = [r for (r,dr,m) in rm] # list with different distances
    difdr = [dr for (r,dr,m) in rm] # list with different distances
    # function to return the neighbor index
    (get_index,nmax) = get_neighbor_order(difdr,sym="C3") # get the index 
    nmat = [[] for i in range(nmax)] # initilize list
    for (r,dr,m) in rm: # loop over matrices of this atom
      nmat[get_index(dr)].append(m) # store this matrix
    mdms = [] # empty list
    print("Found ",len(nmat[1]),"matrices for ",ij)
    for i1 in range(nmax): # loop over distances
# calculate all the averages
      mdms.append(average_matrices(nmat[i1])) 
    dms = [dm[1] for dm in mdms] # store errors
    ms = [im[0] for im in mdms] # store matrices
    if np.max(dms)>0.0001: print(ij,"is not symmetric",np.max(dms))
    else: print(ij,"is symmetric")
    tdict[ij] = (get_index,ms) # store in the dictionary
  def get_symmetric_hopping(i,j,r):
    """Function that return a hopping that respects the crystal symmetries"""
    (gi,ms) = tdict[(i,j)] # get the right set
    try: return ms[gi(r)] # return the right matrix
    except: return None # return None
  return get_symmetric_hopping # return the function
    
    


def get_neighbor_order(drs,sym="C3",prec=1):
  """Returns a function that given an input distance, says the order
  of enighbor it is"""
  delta = 10**(-prec) # precision
  if sym=="C3":  cn = 3 # group symmetry used
  elif sym=="C6":  cn = 6 # group symmetry used
  elif sym=="C4":  cn = 4 # group symmetry used
  alpha = 2.*np.pi/cn # C3 rotation
  sa,ca = np.sin(alpha), np.cos(alpha) # sin and cosine
  Rot = np.matrix([[ca,sa,0.],[-sa,ca,0.],[0.,0.,1.]]) # rotation matrix
  Rs = [Rot**i for i in range(cn)] # rotation matrices
  def same_vectors(dr,rs):
    """Check if two vectors are related by a symmetry operation"""
    norm1 = np.sqrt(dr.dot(dr)) # first norm
    ur1 = dr/norm1 # unit vector
    norm2 = np.sqrt(rs.dot(rs)) # secon norm
    if norm1==0. and norm2==0.: return True # special case of 0 vector
    if not np.abs(norm1-norm2)<delta: return False # different distance
    for R in Rs: # loop over operations
      rs2 = R*(np.matrix([rs]).T) # rotate    
      rs2 = np.array(rs2).reshape(3) # convert to 1d array
      diff = dr - rs2 # vector difference
      if diff.dot(diff)<delta:
#        print("Same",dr-rs)
        return True # this vector in not independent    
    return False # the vectors are different
  out = [] # irreducible vectors
  dis = [] # distances
  for dr in drs: # loop over distances
    if not sum([same_vectors(dr,rs) for rs in out]): # vector not present 
      out.append(dr) # store this vector
      dis.append(dr.dot(dr)) # store this vector
  out = [o.tolist() for o in out] # convert to list to avoid bug
  out = [o for (d,o) in sorted(zip(dis,out))]
  out = [np.array(o) for o in out] # convert back to array
#'  print(sorted(dis))
#  out = zip(out,dis).sort(key=lambda (x,y): b.index(x))
  def get_group(r):
    """Return the group of neighbor"""
    for i in range(len(out)):
      if same_vectors(r,out[i]): return i # group of the vector    
    print("Not found",r,out)
    raise # raise error if not found
  return (get_group,len(out)) # return the function













def average_matrices(ms):
  """Calculate the average of certain matrices"""
  if len(ms)==0: return (None,0.)
  mo = 0.0*ms[0] # initialize
  for m in ms: mo += m # add matrix
  mo = mo/len(ms) # average
  dm = sum([np.sum(np.abs(m-mo)) for m in ms])/len(ms) # error
#  dm /= np.sum(np.abs(ms)) # normalize to the dimension
  return mo,dm




