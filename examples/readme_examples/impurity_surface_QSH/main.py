# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")



from pyqula import geometry
import numpy as np
from pyqula import embedding
g = geometry.honeycomb_lattice() # create geometry of a chain
g = geometry.honeycomb_lattice_armchair() # create geometry of a chain
#g = geometry.honeycomb_lattice_zigzag() # create geometry of a chain
g = g.get_supercell(4)
h = g.get_hamiltonian(has_spin=False)
h.add_haldane(.1)
def fhole(r):
    if r[1]>np.mean(g.r[:,1]):
        if abs(r[0])<4.5: return 0
    return 0.
h.geometry.write_profile(fhole)
hv = h.copy() # copy Hamiltonian to create a defective one
hv.add_onsite(fhole) # add a defect
eb = embedding.Embedding(h,m=hv) # create an embedding object
eb.mode = "surface" # surface mode
(x,y,d) = eb.get_ldos(energy=0.0,delta=1e-2)
fo = open("LDOS.OUT","w")
for (ix,iy,di) in zip(x,y,d):
    if fhole([ix,iy,0.])<1.: 
        fo.write(str(ix)+"  ")
        fo.write(str(iy)+"  ")
        fo.write(str(di)+"\n")
fo.close()
#np.savetxt("LDOS.OUT",np.array([x,y,d]).T)







