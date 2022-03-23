# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
import numpy as np
# create the hamiltonian
g = geometry.triangular_lattice() # triangular lattice geometry
h0 = g.get_hamiltonian() # create hamiltonian of the system


# Now compute energies for different rotations
# mesh of quvectors for the spin spiral
qs = np.linspace(-1,1,20) # loop over qvectors
from pyqula import klist
qpath = klist.default(h0.geometry,nk=100) # default qpath

fo = open("STIFFNESS.OUT","w") # open file
for iq in range(len(qpath)): # loop over qy
    q = qpath[iq] # q-vector
    # Compute the energy by rotating the previous ground state
    ############################
    h = h0.copy() # copy SCF Hamiltonian
    # create the qvector of the rotation
    # get the vector in natural units (assume input is in the real BZ)
    # This is the direction around which we rotate the magnetization
    vector = [0.,0.,1.]
    # rotate the Hamiltonian
    h.generate_spin_spiral(vector=vector,qspiral=q,
            fractional=True)
    h.add_zeeman([4.,0.,0.0])
    # compute the total energy by summing eigenvalues
    e = h.total_energy(nk=20,mode="mesh") # energy computed by rotation
    ############################
    # write in a file
    fo.write(str(iq)+"    "+str(e)+"\n") # write
    print("Doing",iq,e)
    fo.flush() # save result
fo.close()








