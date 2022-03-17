# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")



from pyqula import geometry
import numpy as np
from pyqula import embedding
g = geometry.chain() # create geometry of a chain
h = g.get_hamiltonian() # get the Hamiltonian
eb = embedding.Embedding(h,m=h) # create the embedding object
eb = eb.get_mean_field_hamiltonian(U=3.0,verbose=2)
