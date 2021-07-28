# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=True)
#h.clean()
#h.add_zeeman(0.2)
h.add_antihaldane(0.1)
#h.add_swave(0.1)
h.get_bands(operator=h.get_operator("sz"))
#h.get_bands(operator=h.get_operator("valley"))







