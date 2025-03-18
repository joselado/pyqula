# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import specialhamiltonian

# generate a film of a 3D QSH
h = specialhamiltonian.QSH3D_film(W=20,soc=0.1) 
(k,e,c) = h.get_bands(operator="surface")

import matplotlib.pyplot as plt

plt.scatter(k,e,c=c)
plt.xticks([]) ; plt.xlabel("momentum") ; plt.ylabel("Energy")
plt.ylim([min(e),max(e)]) ; plt.xlim([min(k),max(k)])

plt.show()








