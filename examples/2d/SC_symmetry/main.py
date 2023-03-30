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

# to get singlet/triplet order parameters
print("d-vector",h.get_average_dvector())
print("absolute triplet",h.extract("absolute_delta",mode="triplet"))
print("absolute singlet",h.extract("absolute_delta",mode="singlet"))
exit()

np.savetxt("DELTA.OUT",np.array(h.extract("deltak",mode="both")).real.T)
np.savetxt("DELTA_SINGLET.OUT",np.array(h.extract("deltak",mode="singlet")).real.T)
np.savetxt("DELTA_TRIPLET.OUT",np.array(h.extract("deltak",mode="triplet")).real.T)






