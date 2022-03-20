from . import multicell
from . import sculpt
from . import sculpt

def build(h,nz=1):
  """Create Hamiltonian of a film from a 3d geometry"""
#  if not h.dimensionality==3: raise
  ho = multicell.supercell(h,nsuper=[1,1,nz],sparse=False,ncut=3)
  ho.dimensionality = 2 # reduce dimensionality
  ho.geometry.dimensionality = 2 # reduce dimensionality
  ho.geometry = sculpt.set_xy_plane(ho.geometry) # put in the xy plane
  hoppings = [] # empty list
  for t in ho.hopping: # loop over hoppings
    if t.dir[2]== 0: hoppings.append(t) # remove in the z direction
  ho.hopping = hoppings # overwrite hoppings
  return ho



def geometry_film(g,nz=1):
  """Create the geometry of a film"""
  g = sculpt.set_xy_plane(g) # put in the xy plane
  go = g.get_supercell([1,1,nz]) # create the supercell
  go.dimensionality = 2 # reduce dimensionality
  go.get_fractional(center=True)
  go.fractional2real()
  return go




def hamiltonian_film(hin,nz=10):
  """Return the Hamiltonian of a film"""
  h = hin.copy() # copy Hamiltonian
  h = multicell.supercell_hamiltonian(h,nsuper=[1,1,nz])
  hopout = [] # list
  for i in range(len(h.hopping)): # loop over hoppings
    if abs(h.hopping[i].dir[2])<0.1: 
      hopout.append(h.hopping[i])
  h.hopping = hopout
  h.dimensionality = 2
  h.geometry.dimensionality = 2
  h.geometry = sculpt.set_xy_plane(h.geometry)
  return h


