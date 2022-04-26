# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





import numpy as np
from pyqula import specialgeometry
n = 7 # this is a parameter that controls the size of the moire unit cell
g = specialgeometry.twisted_bilayer(n) # get the geometry of the system
# g.r is the list with the positions in the unit cell
# g.a1 is the first lattice vector
# g.a2 is the second lattice vector
# This function will create hoppings in the structure
ti = 0.4 # this is the interlayer hopping (in terms of the intralayer)
from pyqula.specialhopping import twisted_matrix

fm = twisted_matrix(ti=ti) # gets two list of positions, returns a matrix
def fm2(rs1,rs2):
    m = fm(rs1,rs2) # original matrix
    rm = rs1 - rs2 # average between two positions
    f = lambda r: 1.0 # dummy function
    diag = np.zeros((len(rs1),len(rs1))) # diagonal matrix
    for i in range(len(rs1)): 
        diag[i,i] = f(rm[i]) # set diagonal element
    return diag@m

h = g.get_hamiltonian(is_sparse=True,has_spin=False,is_multicell=True,
     mgenerator=fm2)
hk = h.get_hk_gen() # get Bloch Hamiltonian generator
# hk is a function that given a k point, returns the Bloch Hamiltonian
# The k points are in the interval [0.,1.]
# This method automatically computes the local density of states
from pyqula import ldos
ldos.ldos(h)
#h.get_ldos(e=0.0) # e is the energy
# This method automatically computes the bands
#h.get_bands(num_bands=20)







