# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import specialhamiltonian
h = specialhamiltonian.decorated_triangular(t1=0.5,t2=.0)
h.set_filling(filling=.5,average=False)
h.write_hopping()
#h.write_onsite()
(k,e) = h.get_bands()
import matplotlib.pyplot as plt

plt.scatter(k,e)
plt.show()







