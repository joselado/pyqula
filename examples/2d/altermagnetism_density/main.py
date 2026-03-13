# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





import numpy as np
from pyqula import specialhamiltonian

h = specialhamiltonian.square_altermagnet(am=1.)

(xs,ys) = h.get_spin_splitting_density(nk=100,delta=1e-1,
        energies=np.linspace(-5.,5.,200))

import matplotlib.pyplot as plt

plt.plot(xs,ys)

plt.show()
