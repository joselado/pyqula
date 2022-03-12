# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import geometry
g = geometry.honeycomb_zigzag_ribbon(20) # create geometry of a zigzag ribbon
h = g.get_hamiltonian(has_spin=True) # create hamiltonian of the system
h.add_antiferromagnetism(lambda r: (r[1]>0)*0.5) # add antiferromagnetism
h.add_onsite(lambda r: (r[1]>0)*0.3) # add chemical potential
h.add_swave(lambda r: (r[1]<0)*0.3) # add superconductivity
(k,e,sz) = h.get_bands(operator="sz") # calculate band structure



