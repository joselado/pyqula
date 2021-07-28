# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import topology
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False)
h.add_sublattice_imbalance(0.6)
from pyqula import topology
topology.parallel.cores = 6

op = h.get_operator("valley",projector=True) # valley operator
# if you want to use your custom operator, it should be of the form
# def fun(A,k=[0.,0.,0.]): return A@Op(k)
# where Op(k) is the k-dependent operator that you want
# afterwards, redefine your operator as
# op = operators.Operator(fun) 


(x1,y1) = topology.write_berry(h)
(x,y) = topology.write_berry(h,operator=op)
import matplotlib.pyplot as plt
plt.plot(x,y,c="blue",label="Valley Berry")
plt.scatter(x1,y1,c="red",label="Berry")
plt.legend()
plt.show()







