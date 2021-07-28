# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
from pyqula import geometry
g = geometry.honeycomb_zigzag_ribbon(10) # create geometry of a zigzag ribbon
h = g.get_hamiltonian(has_spin=True) # create hamiltonian of the system
def faf(r):
    if r[1]>0.0: return 0.5
    else: return 0.0
def fsc(r):
    if r[1]<0.0:  return 0.3
    else: return 0.0
h.add_antiferromagnetism(faf)
h.add_swave(fsc)
#h.get_bands(operator="valley") # calculate band structure
h.get_bands() # calculate band structure







