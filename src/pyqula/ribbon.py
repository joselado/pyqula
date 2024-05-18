import numpy as np
from . import supercell
from . import sculpt


def bulk2ribbon(obj,**kwargs):
    from .geometry import Geometry
    from .hamiltonians import Hamiltonian
    if type(obj)==Geometry:
        return geometry_bulk2ribbon(obj,**kwargs)
    elif type(obj)==Hamiltonian:
        return hamiltonian_ribbon(obj,**kwargs)
    else: raise



def geometry_bulk2ribbon(g,n=10,boundary=[1,0],clean=True):
  """Return the geometry of a ribbon"""
  go = g.copy() # copy
  m = [[boundary[0],boundary[1],0],[0,1,0],[0,0,1]] # supercell
  if boundary[0]!=1 or boundary[1]!=0:
    go = supercell.non_orthogonal_supercell(go,m,mode="fill") 
  go = go.supercell((1,n)) # create a supercell
  go = sculpt.rotate_a2b(go,go.a1,np.array([1.,0.,0.]))
  go.dimensionality = 1 # zero dimensional
  go.a2 = np.array([0.,1.,0.])
  if clean: 
    go = go.clean(iterative=True)
    if len(go.r)==0:
      print("Ribbon is not wide enough")
      raise
  go.real2fractional()
  go.fractional2real()
  go.celldis = go.a1[0]
  go.center()
  return go







def hamiltonian_ribbon(hin,n=10):
  """Return the Hamiltonian of a film"""
  h = hin.copy() # copy Hamiltonian
  from . import multicell
  h = multicell.supercell_hamiltonian(h,nsuper=[1,n,1])
  hopout = [] # list
  for i in range(len(h.hopping)): # loop over hoppings
    if abs(h.hopping[i].dir[2])<0.01: 
      if abs(h.hopping[i].dir[1])<0.01: 
        hopout.append(h.hopping[i])
  h.hopping = hopout
  if len(hopout)==0: raise # no hopping found
  h.dimensionality = 1
  h.geometry.dimensionality = 1
  h.geometry = sculpt.rotate_a2b(h.geometry,h.geometry.a1,np.array([1.,0.,0.]))
  return h



def island2ribbon(g):
    """Transform the geometry of an island into a ribbon,
    by doing the minimal possible coupling"""
    if g.dimensionality != 0: raise # only for 1D
    r = g.r.copy() # array with positions
    # as first attempt, do a full shift
    xmin = np.min(g.r[:,0])
    xmax = np.max(g.r[:,0])
    dx = xmax - xmin + 1.0 # maximum shift
    a1 = [dx,0.,0.] # distance
    r2 = [ri + a1 for ri in r]
    from .neighbor import find_first_neighbor
    out = find_first_neighbor(r,r2)
    if len(out)==0: 
        print("island2ribbon implementation not working")
        raise
    # a more general implementation should be included
    go = g.copy()
    go.dimensionality = 1
    go.a1 = np.array([dx,0.,0.])
    return go





