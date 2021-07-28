# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





import numpy as np
from pyqula import specialhamiltonian

from pyqula import topology
#raise # this does not work yet


h = specialhamiltonian.tbg(n=9,ti=0.5)
h.add_onsite(lambda r: 0.1*np.sign(r[2]))
h.set_filling(0.5+3./h.intra.shape[0],nk=1) # set filling
op = h.get_operator("valley",projector=True) # valley operator
#h.get_bands(num_bands=30,operator="valley")
#exit()
#exit()
topology.berry_green_map(h,k=[0.0,0.0,0.0],nrep=5,
        integral=False,operator=op,emin=-0.1)








