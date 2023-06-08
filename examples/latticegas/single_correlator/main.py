# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import geometry
from pyqula import supercell
from pyqula import latticegas
import numpy as np
g = geometry.triangular_lattice() # generate the geometry
g = supercell.turn_orthorhombic(g) # make a orthorombic cell
g = g.get_supercell(10)
g.dimensionality
csa = []
nit = 20
lg = latticegas.LatticeGas(g,filling=1./3.)
#lg.add_interaction(Jij=[1.,1.,.0])
#lg.optimize_energy(temp=0.1,ntries=1e5)
(rs,cs) = lg.get_correlator()
print(cs)
lg.den = 1. - lg.den
(rs,cs) = lg.get_correlator()
print(cs)

