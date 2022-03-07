# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





import neighbor
import multiterminal
import numpy as np
from pyqula import geometry
import sculpt
import pseudocontact
g = geometry.honeycomb_lattice()
imfile = "color_island.png"
imfile = "contact.png"
#imfile = "left_up.png"
g0 = sculpt.image2island(imfile,g,size=20,color="black")
gc = sculpt.image2island(imfile,g,size=20,color="red")
g = sculpt.add(g0,gc)
#g.write()
g.clean()
g = g.clean() # remove single bonds
h = g.get_hamiltonian(has_spin=False,is_sparse=True)
h.add_peierls(0.02)
indexes = sculpt.common(g,gc)
pseudocontact.write_correlator(h,index=indexes,e=0.4)







