# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import ldos
import numpy as np

n = 40
w = n/2. # cutoff for the hopping
g = geometry.chain(n) # chain
g.dimensionality = 0


def ft(n):
    def f0(i1,i2):
        if abs(i1-i2)==1:
            i0 = (i1+i2)/2.0/n # average
            return np.tanh(4.*i0**2)
        else: return 0.
    return f0 # return function


f0 = ft(n) # function

def f(r1,r2):
    i1 = g.get_index(r1)
    i2 = g.get_index(r2)
    return f0(i1,i2)

h = g.get_hamiltonian(fun=f,has_spin=False)
#h = g.get_hamiltonian(has_spin=False)
ldos.multi_ldos(h,delta=2e-2,es=np.linspace(-0.5,0.5,300))








