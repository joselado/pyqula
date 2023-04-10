# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")



from pyqula import geometry
from pyqula.specialhamiltonian import H2HFH
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian()
h.add_onsite(1.0)
h = H2HFH(h,JK=0.2)
(k,e,c) = h.get_bands(operator="dispersive_electrons")
h.get_ldos(operator="electron")






