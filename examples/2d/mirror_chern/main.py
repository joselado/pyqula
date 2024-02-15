# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


import numpy as np
from pyqula import specialhamiltonian
# Hamiltonian
h = specialhamiltonian.multilayer_graphene(l=[0,1],ti=0.0)
# add opposite Haldane couplings to each layer
#h.add_haldane(0.1)
h.add_haldane(lambda r: 0.1*np.sign(r[2]))

kpath = h.geometry.get_kpath(["G","K","M2","K'","G"],nk=80)
(ks,es) = h.get_bands(kpath=kpath)

# define the operator taking  +1 for 1 sector and -1 for the other sector
op = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]]) 
from pyqula.operators import Operator
op = Operator(op)

# compute the operator Berry curvature
be = h.get_berry_curvature(kpath=kpath,operator=op)[2]

# compute the conventional Berry curvature
be0 = h.get_berry_curvature(kpath=kpath)[2]

import matplotlib.pyplot as plt

fig = plt.figure(figsize=[8,8])

plt.subplot(3,1,1)
plt.title("Band structure")
plt.scatter(ks,es) ; plt.xticks([]) ;  plt.ylabel("Energy") ; plt.xlabel("kpath")

plt.subplot(3,1,2)
plt.title("Operator Berry curvature")
plt.scatter(range(len(kpath)),be)
plt.xticks([]) ;  plt.ylabel("Operator Berry") ; plt.xlabel("kpath")

plt.subplot(3,1,3)
plt.title("Berry curvature")
plt.scatter(range(len(kpath)),be0) ; plt.ylim([-1,1])
plt.xticks([]) ;  plt.ylabel("Berry") ; plt.xlabel("kpath")

plt.tight_layout()

plt.show()
