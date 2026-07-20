# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


import numpy as np
from pyqula import geometry,meanfield
def get():
    g = geometry.chain()
    h = g.get_hamiltonian() # create hamiltonian of the system
    J = np.random.random(3) - 0.5
    J = J/np.sqrt(J.dot(J))
    print("Exchange",J)
    J = J*2
    h.add_zeeman(J)
    h.add_swave(0.1)
    scf = meanfield.Vinteraction(h,V1=-1.0,nk=10,filling=0.1,mf="random",
        constrains = ["no_normal_term"],
        verbosity=1)
    triplet = scf.order_parameter("odd_SC")
    singlet = scf.order_parameter("even_SC")
    print("Triplet",triplet)
    print("Singlet",singlet)
    d = scf.hamiltonian.get_average_dvector()
    du = scf.hamiltonian.get_average_dvector(non_unitarity=True)
    print("Average d-vector",sum(d))
#    print("Average non-unitarity of the d-vector",sum(du))
    du = scf.hamiltonian.get_dvector_non_unitarity()[0]
    du = du/np.sqrt(du.dot(du))
    print("Non-unitarity",np.round(du,3))
    return triplet,singlet


triplets,singlets = [],[]
for i in range(2):
    print()
    print("Trial")
    t,s = get()
    triplets.append(t) ; singlets.append(s)

import matplotlib.pyplot as plt

plt.scatter(range(len(triplets)),triplets,label="Triplet")
plt.scatter(range(len(singlets)),singlets,label="Singlet")
plt.xlabel("Trial") ; plt.ylabel("Order parameter")
plt.legend()
plt.show()



