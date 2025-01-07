# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")




# model for a one-dimensional topological heavy-fermion moodel
from pyqula import specialhamiltonian
# optional parameters are te=1.0,tl=0.2,tk=0.3
h = specialhamiltonian.topological_heavy_fermion_1d()
(k,e) = h.get_bands() # calculate band structure

import matplotlib.pyplot as plt

plt.scatter(k,e)

plt.show()








