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
ii = g.get_central()[0]
#print(ii)
#g = g.remove(i=ii)
g.write()
h = g.get_hamiltonian()
#h.add_kekule(.1)
#h.add_sublattice_imbalance(lambda r: 0.6*np.sign(r[2]))
h.add_antiferromagnetism(lambda r: 0.8*np.sign(r[2]))
#h.add_antiferromagnetism(lambda r: 0.8*(r[2]>0))
#h.add_swave(lambda r: 0.8*(r[2]<0))
#h.add_swave(0.0)
mf = meanfield.guess(h,"kanemele")
#mf = meanfield.guess(h,"random")
h.add_kane_mele(0.01)
mf = None
scf = meanfield.Vinteraction(h,V1=3.0,U=0.0,mf=mf,V2=3.0,V3=0.0,nk=20,filling=0.5,mix=0.2,compute_dd=False)
mix0 = h.extract("spin_mixing")
print("Mixing in original Hamiltonian",h.extract("spin_mixing"))
h = scf.hamiltonian
print("Mixing in new Hamiltonian",h.extract("spin_mixing"))
mix1 = h.extract("spin_mixing")
print("Enhancement of the spin mixing",mix1/mix0)
#h.add_rashba(.1)
#h.add_antiferromagnetism(lambda r: 0.6*(r[2]>0))
#h.add_swave(lambda r: 0.6*(r[2]<0))
#print(h.extract("swave"))
print("Density",h.extract("density"))
#print(h.extract("mx"))
#print(h.extract("my"))
print("Mz",h.extract("mz"))
#h.add_zeeman([.0,.0,.1])
#h.write_onsite()
op = h.get_operator("sz")#*h.get_operator("sublattice")
#op = h.get_operator("valley")#*h.get_operator("sublattice")
h.get_bands(operator=op)
#h.write_hopping()
#h.write_swave()
#h.write_magnetization()







