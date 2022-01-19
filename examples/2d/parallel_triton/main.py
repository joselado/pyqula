# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")




import numpy as np

### WARNING, this script only works in a special cluster! (Triton) ###

def func_to_parallelize(U):
    """Function to parallelize"""
    # all the variables must be internal!
    from pyqula import geometry
    from pyqula import meanfield
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian() # create hamiltonian of the system
    return 
    h.add_antiferromagnetism(U)
    return h.get_gap()

from pyqula import parallelslurm

Us = np.linspace(0.,3.0,10) # 10 different calculations
gs = parallelslurm.pcall(func_to_parallelize,Us,time=0.1) # compute for all the inputs
np.savetxt("SWEEP.OUT",np.array([Us,gs]).T) # write in a file







