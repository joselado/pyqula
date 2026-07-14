# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import geometry
g = geometry.bichain() # geometry of the chain
g = g.get_supercell(8) # make a supercell
g.dimensionality = 0
h = g.get_hamiltonian() # get the Hamiltonian
h.add_exchange([0.,1e-2,0.]) # this helps getting the peak at zero
h = h.get_mean_field_hamiltonian(U=3.0,filling=0.5,mf="random") # perform SCF
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.size'] = 10

delta = 5e-2
# compute all the LDOS
energies = np.linspace(0.,3.,100)
import time
t0 = time.time()
iets = h.get_iets_ldos(e=energies,delta=delta)[1] # all the IETS
t1 = time.time()
print("Time in IETS",t1-t0)
ldos = [h.get_ldos(e=e,delta=delta)[2] for e in energies] # all the LDOS

x = g.r[:,0]
fig = plt.figure(figsize=(8,6))
plt.subplot(1,2,1) ; plt.title("LDOS")
plt.contourf(x,energies,np.sqrt(ldos),levels=40,cmap="Blues_r")
plt.xlabel("Site") ; plt.ylabel("Energy")
plt.colorbar(ticks=[],orientation="horizontal",label="dIdV")

plt.subplot(1,2,2) ; plt.title("IETS")
plt.contourf(x,energies,np.sqrt(np.abs(iets)),levels=40,cmap="inferno")
plt.xlabel("Site") ; plt.ylabel("Energy")
plt.colorbar(ticks=[],orientation="horizontal",label="d2IdV2")
plt.tight_layout()

plt.savefig("IETS_chain.png",dpi=500)
plt.show()
