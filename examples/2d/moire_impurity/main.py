# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
import numpy as np

g = geometry.triangular_lattice() # create a triangular lattice
g = g.supercell(5) # create a supercell for the moire

# print the lattice vectors
# by definition, ai*bj = delta_{ij}
print("Lattice vectors")
print("A1 = ",g.a1)
print("A2 = ",g.a2)


print("Reciprocal lattice vectors")
print("B1 = ",g.b1)
print("B2 = ",g.b2)

h = g.get_hamiltonian(has_spin=True) # generate Hamiltonian

# first, lets add a moire potential

### Begin the moire potential ###

# now create a profile commensurate with the unit cell
k1,k2,k3 = g.b1,g.b2,g.b2-g.b1 # wavevectors for the potential

def potential(r):
    """Potential commensurate with the unit cell"""
    v = 0.3 # strength of the potential
    p = 2*np.pi # two pi factor
    vk = lambda k: v*np.cos(k.dot(r)*np.pi*2) # function for one contribution
    return vk(k1) + vk(k2) + vk(k3) # evaluation of the three contributions

# add the moire in the onsite energies
# (you may want to put it in any other parameter)
h.add_onsite(potential) # add the potential to the Hamiltonian

# as a sanity check, write the potential in file to visualize it
g3 = g.supercell(3) # make a supercell for visualization
vi = [potential(ri) for ri in g3.r] # potential in each site
np.savetxt("POTENTIAL.OUT",np.array([g3.r[:,0],g3.r[:,1],vi]).T)

### End the moire potential ###

# now lets add the impurity

# define a function for the impurity
def impurity(r):
  r0 = g.r[0] # location of the impurity
  dr = r - r0
  if dr.dot(dr)<1e-4: # if this is the impurity site
      return 6.0 # large potential
  else: return 0.0


# add the impurity in the onsite energies
h.add_onsite(impurity) # add the impurity potential

h.get_bands()







