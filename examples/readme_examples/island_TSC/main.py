# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import islands
import numpy as np
g = islands.get_geometry(name="triangular",shape="flower",
                           r=14.2,dr=2.0,nedges=6) # get a flower-shaped island
#g.write() ; exit()
h = g.get_hamiltonian() # get the Hamiltonian
h.add_onsite(3.0) # shift chemical potential
h.add_rashba(1.0) # Rashba spin-orbit coupling
h.add_zeeman([0.,0.,0.6]) # Zeeman field
h.add_swave(.3) # add superconductivity
h.get_ldos(e=0.0,num_bands=10) # Spatially resolved DOS







