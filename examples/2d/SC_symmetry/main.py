# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


import numpy as np

from pyqula import geometry
g = geometry.triangular_lattice()
h = g.get_hamiltonian()
h.add_pairing(mode="nodal_fwave",delta=0.1j,d=[0,0,1])
#h.add_pairing(mode="extended_swave",delta=0.4)
h.add_pairing(mode="swave",delta=0.5)
h.get_bands() ; exit()
h0 = h.copy() ; h0.remove_nambu() ; h0.setup_nambu_spinor()

h = h - h0

hk = h.get_hk_gen() # function generating Bloch Hamiltonian

k = np.random.random(3)
print("At k")
print(np.round(hk(k),2))
print("At -k")
print(np.round(hk(-k),2))




exit()
np.savetxt("DELTA.OUT",np.array(h.extract("deltak",mode="both")).real.T)
np.savetxt("DELTA_SINGLET.OUT",np.array(h.extract("deltak",mode="singlet")).real.T)
np.savetxt("DELTA_TRIPLET.OUT",np.array(h.extract("deltak",mode="triplet")).real.T)






