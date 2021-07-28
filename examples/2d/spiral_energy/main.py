# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import scftypes
import numpy as np
# create the hamiltonian
g = geometry.triangular_lattice() # triangular lattice geometry
#g = geometry.chain()
g = g.supercell(2)
h0 = g.get_hamiltonian(has_spin=True) # create hamiltonian of the system
###################

# perform the SCF calculation
#mf = scftypes.guess(h0,"ferro",fun=[1.,0.,0.]) # in-plane guess
#scf = scftypes.selfconsistency(h0,filling=0.5,nkp=20,g=10.0,
#           mf=mf,mix=0.8,maxerror=1e-6)
#hscf = scf.hamiltonian # save the selfconsistent Hamiltonian
#hscf.get_bands(operator="sz") # compute the SCF bandstructure
#########################

hscf = h0.copy() 

# Now compute energies for different rotations
# mesh of quvectors for the spin spiral
qs = np.linspace(-1,1,20) # loop over qvectors
from pyqula import klist
qpath = klist.default(hscf.geometry,nk=100) # default qpath

fo = open("STIFFNESS.OUT","w") # open file
for iq in range(len(qpath)): # loop over qy
    q = qpath[iq] # q-vector
    # Compute the energy by rotating the previous ground state
    ############################
    h = hscf.copy() # copy SCF Hamiltonian
    # create the qvector of the rotation
    # get the vector in natural units (assume input is in the real BZ)
    # This is the direction around which we rotate the magnetization
    vector = [0.,0.,1.]
    # rotate the Hamiltonian

    h.generate_spin_spiral(vector=[0.,0.,1.],qspiral=q,
            fractional=True)
    h.add_zeeman([4.,0.,0.0])
    # compute the total energy by summing eigenvalues
    e = h.total_energy(nk=5,mode="mesh") # energy computed by rotation
    ############################
    # write in a file
    fo.write(str(iq)+"    "+str(e)+"\n") # write
    print("Doing",iq,e)
    fo.flush() # save result
fo.close()








