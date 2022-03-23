# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





import matplotlib.pyplot as plt
from pyqula import geometry
import numpy as np
# create the hamiltonian

ts = [[1.0,0.0],[1.0,0.2],[1.0,0.38],[1.0,0.4],[1.0,0.6]]

for t in ts:
    g = geometry.triangular_lattice() # triangular lattice geometry
    h0 = g.get_hamiltonian(ts=t) # create hamiltonian of the system
    
    
    # Now compute energies for different rotations
    # mesh of quvectors for the spin spiral
    from pyqula import klist
    qpath = klist.default(h0.geometry,nk=100) # default qpath
    qs = np.linspace(0.,1.,len(qpath)) # loop over qvectors
    
    es = []
    
    #fo = open("STIFFNESS.OUT","w") # open file
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
#        h = h.get_mean_field_hamiltonian(U=10.0,Vs=[1.0,0.4,0.2])
        h.add_zeeman([10.,0.,0.0])
        # compute the total energy by summing eigenvalues
        e = h.total_energy(nk=10,mode="mesh") # energy computed by rotation
        es.append(e)
        ############################
        # write in a file
    #    fo.write(str(iq)+"    "+str(e)+"\n") # write
        print("Doing",iq,e)
    #    fo.flush() # save result
    #fo.close()
    
    
    plt.plot(qs,es,label=str(t[1]))


plt.legend()
plt.show()








