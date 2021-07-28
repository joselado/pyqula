# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





import numpy as np
from pyqula import geometry
from pyqula import films
from pyqula import meanfield


g = geometry.diamond_lattice()
g = films.geometry_film(g,nz=4)
#g = g.supercell(3)
h = g.get_hamiltonian()
#h.add_antiferromagnetism(lambda r: 0.6*(r[2]>0))
#h.add_swave(lambda r: 0.6*(r[2]<0))
h.add_antiferromagnetism(lambda r: 0.8*np.sign(r[2]))
#h.add_swave(lambda r: 0.2*r[2]>0.0)
#h.add_swave(0.0)
mf = meanfield.guess(h,"random")
#h.add_kane_mele(0.04)
#mf = None
scf = meanfield.Vinteraction(h,V1=3.0,U=0.0,mf=mf,V2=3.0,V3=0.0,
        nk=6,filling=0.5,mix=0.3,compute_normal=True,compute_dd=False,
        compute_anomalous=False)
h = scf.hamiltonian
print(scf.identify_symmetry_breaking())
h.get_bands(operator="sz")










