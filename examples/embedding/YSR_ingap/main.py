# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")




from pyqula import geometry
import numpy as np
from pyqula import embedding
g = geometry.triangular_lattice() # create geometry
g = g.get_supercell(2) # supercell
h = g.get_hamiltonian() # get the Hamiltonian,spinless
# create a new intraterm, vacancy is modeled as a large onsite potential
h.add_onsite(1.0)
# put a vacancy
r0 = g.r[g.get_central()][0]

def fm(r):
    dr = r-r0 ; dr = dr.dot(dr)
    if dr<0.1: return [.0,.0,4.0] # exchange
    return [0.,0.,0.]

h.add_swave(0.4)
hv = h.copy()
hv.add_zeeman(fm)
vintra = hv.intra
es = np.linspace(-0.3,0.3,200)

from pyqula import parallel
parallel.cores = 6

eb = embedding.Embedding(h,m=hv)

ei = eb.get_energy_ingap_state() # get energy of the impurity state
#ei = 0.
print("Ingap state",ei)
#(x,y,d) = eb.get_ldos(nsuper=3,energy=ei,delta=1e-2) # get data
(x,y,d) = eb.get_didv(nsuper=7,T=0.3,energy=ei,delta=1e-2) # get data
#(x,y,d) = eb.get_kappa(nsuper=3,T=0.3,energy=ei,delta=1e-2) # get data









