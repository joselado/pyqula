# routines to work with wannier hamiltonians
from __future__ import print_function
import numpy as np
from . import hamiltonians
from . import geometry
from scipy.sparse import csc_matrix as csc
from scipy.sparse import bmat
from scipy.sparse import coo_matrix
from . import multicell
from . import angular
import scipy.linalg as lg
from copy import deepcopy
import os
from . import filesystem as fs


def convert(inname="wannier.win",name="wannier"):
  """Read a wannier.win file, and create a folder with
  all the necessary files that characterize the Hamiltonian"""
  fs.rmdir(name) # remove folder
  fs.mkdir(name) # create folder





def read_geometry(input_file="wannier.win"):
  """Reads the geometry of the wannier calculation"""
  ll = read_between("begin unit_cell_cart","end unit_cell_cart",input_file)
  a1 = ll[1].split()
  a2 = ll[2].split()
  a3 = ll[3].split()
  # read the unit vectors
  a1 = np.array([float(a1[0]),float(a1[1]),float(a1[2])]) # first vector
  a2 = np.array([float(a2[0]),float(a2[1]),float(a2[2])]) # second vector
  a3 = np.array([float(a3[0]),float(a3[1]),float(a3[2])]) # second vector
  g = geometry.Geometry()
  g.dimensionality = 2
  g.has_spin = False
  g.has_sublattice = False
  g.a1 = a1  # store vector
  g.a2 = a2  # store vector
  g.a3 = a3  # store vector
  # read the coordinates 
  ll = read_between("begin projections","end projections",input_file)
  rs = [] # empty list for positions
  g.atoms_have_names = True # Atoms have names
  g.atoms_names = [] # initalize
  for l in ll:
    name = l.split(":")[0] # get name of the atom
    r = get_positions(name,input_file) # get positins of the atoms
    g.atoms_names += [name]*len(r) # append this names
    for i in range(len(r)): # to real coordinates
      r[i] = r[i][0]*a1 + r[i][1]*a2 + r[i][2]*a3
    rs += r # store positions
  g.r = np.array(rs) # store in the class
  g.r2xyz() # store also in xyz atributes
  g.get_fractional() # fractional coordinates
  return g


def get_positions(atom,input_file):
  """Get positions of certain orbitals"""
  ll = read_between("begin atoms_frac","end atoms_frac",input_file)
  rs = [] # empty list
  for l in ll: # loop over lines
    l = l.split()
    name = l[0]
    if atom==name: # found atom
      r = np.array([float(l[1]),float(l[2]),float(l[3])]) # position
      rs.append(r) # add to the list
  return rs # return positions


def get_projected_atom_names(input_file):
  """Get the names of all the atoms who have projections"""
  lorb = read_between("begin projections","end projections",input_file)
  names = []
  for l in lorb:
    name = l.split(":")[0]
    if name not in names:
      names.append(name) # add this name
  return names


def get_orbitals(specie,input_file="wannier.win"):
  """ Get the names of the orbitals of a cetain atom"""
  lorb = read_between("begin projections","end projections",input_file)
  orbs = []
  for l in lorb: # loop over orbitals
    sname = l.split(":")[0] # name of the atom
    oname = l.split(":")[1] # name of the orbital
    oname = oname.split()[0] # remove the \n
    if specie==sname: orbs.append(oname)
  return orbs

def get_all_orbitals(input_file="wannier.win"):
  """Get the name of all the orbitals"""
  atoms = get_projected_atom_names(input_file)
  nout = []
  for a in atoms:
    for io in get_orbitals(a,input_file=input_file): # loop over orbitals
      nat = get_atoms_specie(a,input_file=input_file) # number of atoms
      nout += [a+str(i)+"," + io for i in range(nat)]
  return nout



def get_atoms_specie(specie,input_file="wannier.win"):
  """ Get number of atoms o a certain specie"""
  ll = read_between("begin atoms_frac","end atoms_frac",input_file)
  nat = 0
  for l in ll:
    name = l.split()[0] # name of the atom
    if name==specie: nat +=1 
  return nat # return umber of atoms






def get_index_orbital(specie,atom,orbital,input_file="wannier.win"):
  """Get the index of a certain orbital"""
  ll = read_between("begin atoms_frac","end atoms_frac",input_file)
  lorb = read_between("begin projections","end projections",input_file)
  iorb = 0 # index in the matrix
  print(specie)
  for l in lorb: # loop over orbitals
    sname = l.split(":")[0] # name of the atom
    oname = l.split(":")[1] # name of the orbital
    oname = oname.split()[0] # remove the \n
    ifound = 0 # number of atoms found
    for la in ll: # loop over atoms in the structure
      rname = la.split()[0] # name of the atom in the structure
      if rname==sname: 
        iorb += 1 # increase de counter of the hamiltonian index 
        ifound += 1 # increase the counter in the atoms
      # if the desired orbital and atom has been reached
      if specie==sname and atom==ifound and orbital==oname:
        return iorb
  raise # error if this point is reached



def get_indexes(input_file="wannier.win"):
  """Returns a list with the indexes of the different orbitals for
  each atom """
  ll = read_between("begin atoms_frac","end atoms_frac",input_file)
  lorb = read_between("begin projections","end projections",input_file)
  # first get all the different names
  names = get_projected_atom_names(input_file) # get names of the atoms
  dat = dict() # create a diccionary
  for name in names: # loop over different atoms
    out = [] # empty list with the different indexes and names
    ind = 0 # initialize index of the hamiltonian
    for l in ll: # loop over atoms in the structure
      out_atom = [] # empty list for the next atom
      l = l.split() # split the line
      label = l[0]  # get label of the atom
      if label==name: # if same name as input
        for lo in lorb: # go to the orbitals list and look for this atom
          name_o = lo.split(":")[0] # get name of the atom in the orbital list
          if name_o==name: # if an orbital of this atom has been found 
            orb = lo.split(":")[1] # name of the orbital
            orb = orb.split()[0] # name of the orbital
            out_atom.append((ind,orb)) # append tuple with index and norm
        out.append([out_atom]) # append this atom to the final 
    dat[name] = out # dd this atom to the diccionary
  return dat  # return the diccionary








def read_between(a,b,input_file):
  ll = open(input_file).readlines()
  start = False # found the klist
  out = []
  for (i,l) in zip(range(len(ll)),ll):
    if b in l: break # end of klist
    if start: # sotre line
      out.append(l)
    if a in l: start = True # found beginning 
  return out # return output lines





def read_hamiltonian(input_file="hr_truncated.dat",is_real=False):
  """Reads an output hamiltonian from wannier"""
  mt = np.genfromtxt(input_file) # get file
  m = mt.transpose() # transpose matrix
  # read the hamiltonian matrices
  class Hopping: pass # create empty class
  tlist = []
  def get_t(i,j,k):
    norb = np.max([np.max(np.abs(m[3])),np.max(np.abs(m[4]))])
    norb = int(norb)
    mo = np.matrix(np.zeros((norb,norb),dtype=np.complex))  
    for l in mt: # look into the file
      if i==int(l[0]) and j==int(l[1]) and k==int(l[2]):
        if is_real:
          mo[int(l[3])-1,int(l[4])-1] = l[5] # store element
        else:
          mo[int(l[3])-1,int(l[4])-1] = l[5] + 1j*l[6] # store element
    return mo # return the matrix
#  for i in range(-nmax,nmax):
#    for j in range(-nmax,nmax):
#      for k in range(-nmax,nmax):
#        t = Hopping() # create hopping
#        t.dir = [i,j,k] # direction
#        t.m = get_t(i,j,k) # read the matrix
#        tlist.append(t) # store hopping
  # the previous is not used yet...
  g = geometry.kagome_lattice() # create geometry
  h = g.get_hamiltonian() # build hamiltonian
  h.intra = get_t(0,0,0)
  h.tx = get_t(1,0,0)
  h.ty = get_t(0,1,0)
  h.txy = get_t(1,1,0)
  h.txmy = get_t(1,-1,0)
  h.has_spin = False  # if not spin polarized
  h.geometry = read_geometry() # read the geometry of the system
  if len(h.geometry.r)!=len(h.intra): 
    print("Dimensions do not match",len(g.r),len(h.intra))
    print(h.geometry.r)
    raise # error if dimensions dont match
  return h



def read_multicell_hamiltonian(input_file="hr_truncated.dat",
                                ncells=None,win_file="wannier.win",
                                dim=2,skip_win=False,path=None):
  """Reads an output hamiltonian from wannier"""
  if path is not None: 
      inipath = os.getcwd() # current path
      os.chdir(path) # go there
  mt = np.genfromtxt(input_file) # get file
  m = mt.transpose() # transpose matrix
  if ncells is None: # use all the hoppings
    nmax = int(np.max([np.max(m[i])for i in range(3)]))
    ncells = [nmax,nmax,nmax]
  # read the hamiltonian matrices
  class Hopping: pass # create empty class
  tlist = []
  def get_t(i,j,k):
    norb = np.max([np.max(np.abs(m[3])),np.max(np.abs(m[4]))])
    norb = int(norb)
    mo = np.matrix(np.zeros((norb,norb),dtype=np.complex))  
    found = False
    for l in mt: # look into the file
      if i==int(l[0]) and j==int(l[1]) and k==int(l[2]):
        mo[int(l[3])-1,int(l[4])-1] = l[5] + 1j*l[6] # store element
        found  = True # matrix has been found
    if found:  return mo # return the matrix
    else: return None # return nothing if not found
  for i in range(-ncells[0],ncells[0]+1):
    for j in range(-ncells[1],ncells[1]+1):
      for k in range(-ncells[2],ncells[2]+1):
        if (i,j,k)==(0,0,0): continue # skip intracell
        matrix = get_t(i,j,k) # read the matrix
        if matrix is None: continue
        else: # store hopping
          t = Hopping() # create hopping
          t.dir = [i,j,k] # direction
          t.m = get_t(i,j,k) # read the matrix
          tlist.append(t) # store hopping
  # the previous is not used yet...
  g = geometry.kagome_lattice() # create geometry
  h = g.get_hamiltonian() # build hamiltonian
  h.is_multicell = True
  if not skip_win: # do not skip wannier.win
    h.orbitals = get_all_orbitals(input_file=win_file)
  h.hopping = tlist # list of hoppings
  h.has_spin = False  # if not spin polarized
  if not skip_win: # do not skip readin wannier.win
    h.geometry = read_geometry(input_file=win_file) # read the geometry of the system
    h.geometry.center() # center the geometry
  h.intra = get_t(0,0,0)
  if not skip_win: # do not skip reading wannier.win
    if len(h.geometry.r)!=len(h.intra): 
      print("Dimensions do not match",len(g.r),len(h.intra))
      print(h.geometry.r)
  #  raise # error if dimensions dont match
  h.dimensionality = dim 
  if path is not None: 
      os.chdir(inipath) # go back
      h.wannierpath = path # store
  else: h.wannierpath = None
  # now lets add the SOC method
  def get_soc(self,name,soc):
      if not self.has_spin: raise # only for spinful
      self.intra = self.intra + generate_soc(name,soc,path=self.wannierpath) 
  import types
  h.get_soc = types.MethodType(get_soc,h) # add the method
  return h







def get_klist(input_file="wannier.win",nkpoints=500):
  """ Get the klist for bands calculation"""
  ll = read_between("begin kpoint_path","end kpoint_path",input_file)
  kp = [] # empty vertex
  for l in ll: 
    l2 = l.split() # split the numbers
    kp.append([[float(l2[1]),float(l2[2]),float(l2[3])],[float(l2[5]),float(l2[6]),float(l2[7])]])
  klist = [] # empty klist
  nk = int(nkpoints/len(kp)) # number of kpoints
  for (k1,k2) in kp: # loop over pairs
    k1 = np.array(k1)
    k2 = np.array(k2)
    dk = (k2-k1)/nk # create dk
    for i in range(nk): # loop over ks
      klist.append(k1+i*dk)
  return klist


def get_num_wannier(input_file):
  """Get the number of wannier orbitals"""
  ll = read_between("begin atoms_frac","end atoms_frac",input_file)
  lorb = read_between("begin projections","end projections",input_file)
  norb = 0
  for o in lorb: # loop over orbitals
    name = o.split(":")[0] # name of the orbital
    for l in ll:
      rname = l.split()[0] # name of the atom
      if name==rname: norb += 1
  return norb # return number of orbitals




def write_wannier_names(input_file="wannier.win",output_file="ORBITALS.OUT"):
  """Get the number of wannier orbitals"""
  ll = read_between("begin atoms_frac","end atoms_frac",input_file)
  lorb = read_between("begin projections","end projections",input_file)
  norb = 0
  fo = open(output_file,"w") # open file to write
  fo.write("# Name of the atom,    index of this type of atom, name of the orbital\n")
  for o in lorb: # loop over orbitals
    iatom = 1 # initialize
    name = o.split(":")[0] # name of the orbital
    oname = o.split(":")[1] # name of the orbital
    oname = oname.split()[0]
    for l in ll:
      rname = l.split()[0] # name of the atom
      if name==rname: # if this atom found
        fo.write(name + "              " +str(iatom)+"               "+oname+"\n") # new line
        iatom += 1 # increase atom counter
  fo.close()
  print("Orbitals written to",output_file)









def get_num_atoms(specie,input_file):
  """Get the number of wannier orbitals"""
  ll = read_between("begin atoms_frac","end atoms_frac",input_file)
  nat = 0 # initialize
  for l in ll:
    name = l.split()[0] # name of the atom
    if name==specie: nat += 1
  return nat # return number of orbitals






def generate_soc(specie,value,input_file="wannier.win",nsuper=1,path=None):
  """Add SOC to a hamiltonian based on wannier.win"""
  if path is not None: 
      inipath = os.getcwd() # current path
      os.chdir(path) # go there
  o = open(".soc.status","w")
  iat = 1 # atom counter
  orbnames = names_soc_orbitals(specie) # get which are the orbitals
  ls = soc_l((len(orbnames)-1)/2) # get the SOC matrix
  norb = get_num_wannier(input_file) # number of wannier orbitals
  m = np.matrix([[0.0j for i in range(norb*2)] for j in range(norb*2)]) # matrix
  nat = get_num_atoms(specie,input_file) # number of atoms of this specie
  for iat in range(nat):
   # try:
#      fo.write("Attempting "+specie+"  "+str(iat+1)+"\n")
      for i in range(len(orbnames)): # loop over one index
        orbi = orbnames[i] 
        for j in range(len(orbnames)): # loop over other index
          orbj = orbnames[j]
          ii = get_index_orbital(specie,iat+1,orbi)  # index in wannier
          jj = get_index_orbital(specie,iat+1,orbj) # index in wannier
#          fo.write(str(ii)+"   "+str(jj)+"\n")
          ii += -1 # python starts in 0
          jj += -1 # python starts in 0
          m[2*ii,2*jj] = ls[2*i,2*j] # store the soc coupling
          m[2*ii+1,2*jj] = ls[2*i+1,2*j] # store the soc coupling
          m[2*ii,2*jj+1] = ls[2*i,2*j+1] # store the soc coupling
          m[2*ii+1,2*jj+1] = ls[2*i+1,2*j+1] # store the soc coupling
   #   return
   # except: break
#  fo.close()
  n = nsuper**2 # supercell
  mo = [[None for i in range(n)] for j in range(n)]
  for i in range(n): mo[i][i] = csc(m) # diagonal elements
  mo = bmat(mo).todense() # dense matrix
  if path is not None: 
      os.chdir(inipath) # go there
      print(np.max(np.abs(mo)))
  return np.matrix(mo*value) # return matrix
#  for name in atoms: # loop over atoms



def names_soc_orbitals(specie):
  return angular.names_soc_orbitals(specie)




def ylm2xyz_l2(has_spin=True):
  """Return the matrix that converts the cartesian into spherical harmonics"""
  m = angular.ylm2xyz_l2() # return change of bassi matrix
  if has_spin:
    from .increase_hilbert import spinful
    m = spinful(m) # with spin degree of freedom
  return m



def ylm2xyz_l1(has_spin=True):
  """Return the matrix that converts the cartesian into spherical harmonics"""
  m = angular.ylm2xyz_l1() # return change of bassi matrix
  if has_spin:
    from .increase_hilbert import spinful
    m = spinful(m) # with spin degree of freedom
  return m







def soc_l(l):
  """Calculate the spin orbit coupling in a basis of spherical harmonics"""
  l = int(l)
  nm = 2*l + 1 # number of components
  zero = np.matrix([[0.0j for i in range(2*nm)] for j in range(2*nm)]) 
  # initialize matrices
  lz = zero.copy()
  lx = zero.copy()
  ly = zero.copy()
  lm = zero.copy()
  lp = zero.copy()
  # create l+ and l- and lz
  for m in range(-l,l): # loop over m components
    val = np.sqrt((l-m)*(l+m+1)) # value of the cupling
    im = m + l
    lp[2*(im+1),2*im] = val # up channel
    lp[2*(im+1)+1,2*im+1] = val # down channel
  for m in range(-l,l+1): # loop over m components
    im = m + l
    lz[2*im,2*im] = m # value of lz, up channel
    lz[2*im+1,2*im+1] = m # value of lz, down channel
  lm = lp.H # adjoint
  lx = (lp + lm) /2.
  ly = -1j*(lp - lm) /2.
  # create spin matrices
  sz = zero.copy()
  sx = zero.copy()
  sy = zero.copy()
  for m in range(-l,l+1): # loop over m components
    im = m + l
    sx[2*im,2*im+1] = 1.0 
    sx[2*im+1,2*im] = 1.0 
    sy[2*im,2*im+1] = -1j 
    sy[2*im+1,2*im] = 1j 
    sz[2*im,2*im] = 1.
    sz[2*im+1,2*im+1] = -1.
  # check that the matrix is fine
  sx = sx/2.
  sy = sy/2.
  sz = sz/2.
  if True:
    comm_zero(sx,lx)
    comm_zero(sx,ly)
    comm_zero(sx,lz)
    comm_zero(sy,ly)
    comm_zero(sz,lz)
    comm_angular(sx,sy,sz)
    comm_angular(sy,sz,sx)
    comm_angular(sz,sx,sy)
    comm_angular(lx,ly,lz)
    comm_angular(ly,lz,lx)
    comm_angular(lz,lx,ly)
  ls = lx*sx + ly*sy + lz*sz  # SOC matrix
  import scipy.linalg as lg
  from scipy.sparse import csc_matrix as csc
  if l==2: R = ylm2xyz_l2() # get change of basis matrix
  if l==1: R = ylm2xyz_l1() # get change of basis matrix
  ls = R.H * ls * R # change to cartesian orbitals
  return ls # return the matrix



def comm_angular(x,y,z):
  from scipy.sparse import csc_matrix as csc
  xy = x*y - y*x
  xy = xy - 1j*z
  if np.abs(np.max(xy))>0.001:
    raise



def comm_zero(x,y):
  xy = x*y - y*x
  if np.abs(np.max(xy))>0.01:
    raise



def symmetrize_atoms(h,specie,input_file="wannier.win"):
  """Symmetrizes a certain atom"""
  orbs = get_orbitals(specie,input_file=input_file) # read the orbitals
  nat = get_atoms_specie(specie,input_file=input_file) # number of atoms
  if h.has_spin: raise
  for iorb in orbs: # loop over orbitals
    avg = 0.
    for iat in range(nat): # loop over atoms 
      i = get_index_orbital(specie,iat+1,iorb) - 1 
      ons = h.intra[i,i] # add to the average
      avg += ons # add to the average
    avg = avg/nat # average value
    for iat in range(nat): # loop over atoms
      i = get_index_orbital(specie,iat+1,iorb) - 1
      h.intra[i,i] = avg # substitute by average


def get_hoppings(h,specie1,specie2,input_file="wannier.win"):
  """Get hoppings between two atoms"""
  orbs1 = get_orbitals(specie1,input_file=input_file) # read the orbitals
  orbs2 = get_orbitals(specie2,input_file=input_file) # read the orbitals
  nat1 = get_atoms_specie(specie1,input_file=input_file) # number of atoms
  nat2 = get_atoms_specie(specie2,input_file=input_file) # number of atoms
  mats = [h.intra,h.tx,h.ty,h.txy,h.txmy]
  for iorb1 in orbs1: # loop over orbitals
    for iorb2 in orbs2: # loop over orbitals
      for iat1 in range(nat1): # loop over atoms 
        for iat2 in range(nat2): # loop over atoms 
          i = get_index_orbital(specie1,iat1+1,iorb1) - 1 
          j = get_index_orbital(specie2,iat2+1,iorb1) - 1 
          for m in mats:
            ons = m[i,j].real # add to the average



def get_atomic_projection(specie,input_file="wannier.win",has_spin=False):
  """Get the matrix that projects onto a certain atom"""
  orbs = get_orbitals(specie,input_file=input_file) # read the orbitals
  nat = get_atoms_specie(specie,input_file=input_file) # number of atoms
  norb = get_num_wannier(input_file) # number of wannier orbitals
  proj = np.matrix([[0.0j for i in range(norb)] for j in range(norb)])
  for iat in range(nat): # loop over atoms of this type
    for iorb in orbs: # loop over orbitals
      i = get_index_orbital(specie,iat+1,iorb) - 1 
      proj[i,i] = 1.0 # non vanishing
  if has_spin:
    from .increase_hilbert import spinful
    return spinful(proj)
  else: return proj





def read_supercell_hamiltonian(input_file="hr_truncated.dat",is_real=False,nsuper=1):
  """Reads an output hamiltonian for a supercell from wannier"""
  mt = np.genfromtxt(input_file) # get file
  m = mt.transpose() # transpose matrix
  # read the hamiltonian matrices
  class Hopping: pass # create empty class
  tlist = []
  def get_t(i,j,k):
    norb = int(np.max([np.max(np.abs(m[3])),np.max(np.abs(m[4]))]))
    mo = np.matrix(np.zeros((norb,norb),dtype=np.complex))  
    for l in mt: # look into the file
      if i==int(l[0]) and j==int(l[1]) and k==int(l[2]):
        if is_real:
          mo[int(l[3])-1,int(l[4])-1] = l[5] # store element
        else:
          mo[int(l[3])-1,int(l[4])-1] = l[5] + 1j*l[6] # store element
    return mo # return the matrix
  # this function will be called in a loop
  g = geometry.kagome_lattice() # create geometry
  h = g.get_hamiltonian() # build hamiltonian
  h.has_spin = False
  nstot = nsuper**2
  intra = [[None for i in range(nstot)] for j in range(nstot)]
  tx = [[None for i in range(nstot)] for j in range(nstot)]
  ty = [[None for i in range(nstot)] for j in range(nstot)]
  txy = [[None for i in range(nstot)] for j in range(nstot)]
  txmy = [[None for i in range(nstot)] for j in range(nstot)]
  from scipy.sparse import csc_matrix as csc
  vecs = []
  # create the identifacion vectors
  inds = []
  acu = 0
  try: # read different supercells
    nsuperx = nsuper[0]
    nsupery = nsuper[1]
    nsuperz = nsuper[2]
  except: # read different supercells
    nsuperx = nsuper
    nsupery = nsuper
    nsuperz = nsuper
  for i in range(nsuperx): # loop over first replica
    for j in range(nsupery): # loop over second replica
      vecs.append(np.array([i,j])) # append vector
      inds.append(acu)
      acu += 1 # add one to the accumulator
  for i in inds: # loop over first vector
    for j in inds:  # loop over second vector
      v1 = vecs[i] # position of i esim cell
      v2 = vecs[j] # position of j esim cell
      dv = v2 - v1 # difference in vector
      # get the different block elements
      intra[i][j] = csc(get_t(dv[0],dv[1],0))
      tx[i][j] = csc(get_t(dv[0]+nsuper,dv[1],0))
      ty[i][j] = csc(get_t(dv[0],dv[1]+nsuper,0))
      txy[i][j] = csc(get_t(dv[0]+nsuper,dv[1]+nsuper,0))
      txmy[i][j] = csc(get_t(dv[0]+nsuper,dv[1]-nsuper,0))
  h.intra = bmat(intra).todense()
  h.tx = bmat(tx).todense()
  h.ty = bmat(ty).todense()
  h.txy = bmat(txy).todense()
  h.txmy = bmat(txmy).todense()
  h.geometry = read_geometry() # read the geometry of the system
  if nsuper>1:
    h.geometry = h.geometry.supercell(nsuper) # create supercell
  if len(h.geometry.r)!=len(h.intra): 
    print("Dimensions do not match",len(g.r),len(h.intra))
    print(h.geometry.r)
 #   raise # error if dimensions dont match
  # names of the orbitals
  h.orbitals = get_all_orbitals()*nsuper**2

  return h




def rotate90(h):
  """ Rotate by 90 degrees th unit cell"""
  h.geometry.x, h.geometry.y = h.geometry.y, h.geometry.x
  h.geometry.xyz2r()
  a1 = h.geometry.a1
  a2 = h.geometry.a2
  h.geometry.a1, h.geometry.a2 = np.array([a2[1],a2[0],0.]),  np.array([a1[1],a1[0],0.])  # geometry
  h.tx , h.ty = h.ty, h.tx
  h.txy , h.tmy = h.txy, h.txmy.H
  return h




def save(output_file="wannier.xml",hamiltonian=None):
  """Saves a wannier hamiltonian in xml format"""
  import xml.etree.cElementTree as ET
  strf = "{:10.8f}".format  # float format
  if hamiltonian is None: # no hamiltonian provided
    hamiltonian = read_multicell_hamiltonian() # read hamiltonian
  else:
    hamiltonian = multicell.turn_multicell(hamiltonian)
  root = ET.Element("wannier") # root element
  try: orbs = hamiltonian.orbitals # name of the orbitals
  except: orbs = ["X,Y" for i in range(hamiltonian.intra.shape[0])]
  geo = hamiltonian.geometry # geometry of the crystal
  #### start write the orbitals ###
  orbxml = ET.SubElement(root, "orbitals") # subelement orbitals
  fo = open("orbitals.wan","w") # human readable
  ffrac = open("fractional_coordinates.wan","w") # human readable
  freal = open("real_coordinates.wan","w") # human readable
  fo.write("# index, name,       position\n")
  ffrac.write("# position\n")
  freal.write("# position\n")
  for i in range(len(geo.r)):
    orbital = ET.SubElement(orbxml, "orbital") # subelement orbital
    ET.SubElement(orbital, "index").text = str(i)
    ET.SubElement(orbital, "atom_name").text = orbs[i].split(",")[0]
    ET.SubElement(orbital, "orbital_name").text = orbs[i].split(",")[1]
    ET.SubElement(orbital, "x").text = strf(geo.x[i])
    ET.SubElement(orbital, "y").text = strf(geo.y[i])
    ET.SubElement(orbital, "z").text = strf(geo.z[i])
    ET.SubElement(orbital, "frac_x").text = strf(geo.frac_x[i])
    ET.SubElement(orbital, "frac_y").text = strf(geo.frac_y[i])
    ET.SubElement(orbital, "frac_z").text = strf(geo.frac_z[i])
    # human readable
    fo.write("    "+str(i+1)+"    "+orbs[i]+"    ")
    fo.write(strf(geo.x[i])+"    "+strf(geo.y[i])+"   "+strf(geo.z[i])+"  ")
    freal.write(strf(geo.x[i])+"    "+strf(geo.y[i])+"   "+strf(geo.z[i])+"\n")
    fo.write(strf(geo.frac_x[i])+"    "+strf(geo.frac_y[i])+"   "+strf(geo.frac_z[i])+"\n")
    ffrac.write(strf(geo.frac_x[i])+"      "+strf(geo.frac_y[i])+"      "+strf(geo.frac_z[i])+"\n")
  #### end write the orbitals ###
  fo.close()
  ffrac.close()
  freal.close()
  #### start write the geometry ###

  geoxml = ET.SubElement(root, "geometry") # subelement orbitals
  aname = ["a1","a2","a3"] # name of unit vectors
  aobj = [geo.a1,geo.a2,geo.a3]
  fg = open("geometry.wan","w") # geometry file
  fg.write("# unit cell vectors\n")
  for (an,ao) in zip(aname,aobj): # loop over vectors
    a1xml = ET.SubElement(geoxml, an) # subelement orbital
    ET.SubElement(a1xml, "x").text = strf(ao[0])
    ET.SubElement(a1xml, "y").text = strf(ao[1])
    ET.SubElement(a1xml, "z").text = strf(ao[2])
    fg.write(strf(ao[0])+"  "+strf(ao[1])+"  "+strf(ao[2])+"\n")
  fg.close()
  #### end write the geometry ###

  #### start write the hamiltonian ###
  
  ham = ET.SubElement(root, "hamiltonian") # subelement hamiltonian
  fo = open("hamiltonian.wan","w") # human readable
  fo.write("# nx, ny, nz, i, j, real, imaginary\n")
  # intracell object (workaround)
  intraobj = deepcopy(hamiltonian.hopping[0])
  intraobj.dir = [0,0,0] # store direction
  intraobj.m = hamiltonian.intra # store matrix
  for hop in [intraobj]+hamiltonian.hopping:
    d = hop.dir # direction
    m = hop.m # hopping matrix
    m = coo_matrix(m) # to coo matrix
    ii = m.row # row index
    jj = m.col # column idex
    data = m.data # data
    for (i,j,mij) in zip(ii,jj,data): # loop over elements
      dist = geo.r[i] - (geo.r[j]+d[0]*geo.a1+d[1]*geo.a2+d[2]*geo.a3)
      dist = np.sqrt(dist.dot(dist))
      element = ET.SubElement(ham, "hopping") # subelement hamiltonian
      ET.SubElement(element, "nx").text = str(d[0]) # element
      ET.SubElement(element, "ny").text = str(d[1]) # element
      ET.SubElement(element, "nz").text = str(d[2]) # element
      ET.SubElement(element, "i").text = str(i+1) # element
      ET.SubElement(element, "j").text = str(j+1) # element
      ET.SubElement(element, "iname").text = orbs[i] # element
      ET.SubElement(element, "jname").text = orbs[j] # element
      ET.SubElement(element, "real_amplitude").text = strf(mij.real) # element
      ET.SubElement(element, "imag_amplitude").text = strf(mij.imag) # element
      ET.SubElement(element, "distance").text = strf(dist) # element

      # human readable
      fo.write("   "+str(d[0])+"   "+str(d[1])+"   "+str(d[2])+"   ") 
      fo.write(str(i+1)+"    "+str(j+1)+"    ")
      fo.write(strf(mij.real)+"    "+strf(mij.imag)+"\n")
  fo.close() # close hamiltonian file
  #### end write the hamiltonian ###


  #### save everything ###
  tree = ET.ElementTree(root) # create whole tree
  tree.write("wannier.xml") # write tree
  import xml.dom.minidom
  xml = xml.dom.minidom.parse("wannier.xml") 
  open("wannier.xml","w").write(xml.toprettyxml())





def save_readable(hamiltonian,output_file="hamiltonian.wan"):
  """Saves a wannier hamiltonian in xml format"""
  strf = "{:10.8f}".format  # float format
  hamiltonian = multicell.turn_multicell(hamiltonian)
  #### start write the hamiltonian ###
  fo = open(output_file,"w") # human readable
  fo.write("# nx, ny, nz, i, j, real, imaginary\n")
  # intracell object (workaround)
  class ObjHop: pass
  intraobj = ObjHop()
  intraobj.dir = [0,0,0] # store direction
  intraobj.m = hamiltonian.intra # store matrix
  for hop in [intraobj]+hamiltonian.hopping:
    d = hop.dir # direction
    m = hop.m # hopping matrix
    m = coo_matrix(m) # to coo matrix
    ii = m.row # row index
    jj = m.col # column idex
    data = m.data # data
    for (i,j,mij) in zip(ii,jj,data): # loop over elements
      # human readable
      fo.write("   "+str(d[0])+"   "+str(d[1])+"   "+str(d[2])+"   ") 
      fo.write(str(i+1)+"    "+str(j+1)+"    ")
      fo.write(strf(mij.real)+"    "+strf(mij.imag)+"\n")
  fo.close() # close hamiltonian file
  #### end write the hamiltonian ###







