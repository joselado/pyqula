# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import topology
from pyqula import spectrum
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=True)
h.add_onsite(0.6)
h.get_bands()
spectrum.fermi_surface(h,nk=100,operator="valley")







