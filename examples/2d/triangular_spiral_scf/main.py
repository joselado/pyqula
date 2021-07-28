# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
# traingular lattice Hamiltonian
g = geometry.triangular_lattice()
g = g.supercell(3) # get a 3x3 supercell
h = g.get_hamiltonian(is_multicell=False,has_spin=False)
# if you wanted to modify the Hamiltonian, just put here your data
# h.geometry.r = rs # new positions
# h.geometry.r2xyz() # update
# h.geometry.a1 = a1 # first lattice vector
# h.geometry.a2 = a2 # second lattice vector
# h.intra = intra # your (spinless) intra cell hopping
# h.tx = tx # your (spinless) (1,0,0) hopping matrix
# h.ty = ty # your (spinless) (0,1,0) hopping matrix
# h.txy = txy # your (spinless) (1,1,0) hopping matrix
# h.txmy = txmy # your (spinless) (1,-1,0) hopping matrix

# now make the Hamiltonian spinful
h.turn_spinful()

# let us do now a mean field calculation
from pyqula import meanfield

mf = meanfield.guess(h,"randomXY") # random initial guess in XY plane
# if you want to restart a calculation, just uncomment the mf=None line
# that will force the code to pick the latest mean field from a file
#mf = None
scf = meanfield.hubbardscf(h,U=6.0,mf=mf,
        verbose=1,filling=0.5,nk=6,mix=0.5) # perform SCF calculation 
# this will write the magnetization to MAGNETISM.OUT
scf.hamiltonian.write_magnetization(nrep=2) # write magnetization in a file









