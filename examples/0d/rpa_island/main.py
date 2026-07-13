# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import islands
g = islands.get_geometry(name="honeycomb",n=2,nedges=3) # get an island
h = g.get_hamiltonian() # get the Hamiltonian
h.add_exchange([0.,0.,1e-2]) # this helps getting the peak at zero
h = h.get_mean_field_hamiltonian(U=1.5,filling=0.5,mf="random") # perform SCF
print("Number of sites",len(h.geometry.r))

import numpy as np
energies=np.linspace(0.,.4,200)

import time
t0 = time.time()
es,chis = h.get_spinchi_full(energies=energies,delta=1e-2) # compute RPA tensor
t1 = time.time()
print("Time in IETS",t1-t0)

mz = h.get_vev("sz")

(es_dos,dos) = h.get_dos(energies=es,delta=1e-2)

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 10

# plot the spin response function
cs = np.array([np.trace(c).imag for c in chis])
#print(cs)
fig = plt.figure(figsize=(10,4))
plt.subplot(1,3,1) ; plt.title("Spin excitations")
plt.plot(es,cs,c="blue")
plt.xlabel("Energy") ; plt.ylabel("Dynamical spin correlator")
plt.xlim([min(es),max(es)]) #; plt.ylim([0.,np.max(cs)*1.1]) ; plt.yticks([])

# now plot the DOS
plt.subplot(1,3,2) ; plt.title("Electronic DOS")
plt.plot(es_dos,dos,c="red")
plt.xlabel("Energy") ; plt.ylabel("electronic DOS")
plt.xlim([min(es),max(es)]) ; plt.ylim([0.,np.max(dos)*1.1]) ; plt.yticks([])

#
# now the simulated dIdV
cs_int = [np.sum(cs[0:i]) for i in range(len(cs))]
cs_int = np.array(cs_int) ; cs_int = cs_int/np.max(cs_int) # as array, and normalize
dos_int = np.array(dos)/np.max(dos)
ratio = 0.5 # ratio between inelastic and elastic contributions
d2idv2 = ratio*cs_int + dos_int

# now plot dIdV with both contributions
plt.subplot(1,3,3) ; plt.title("Simulated dIdV")
cols = dos_int/d2idv2 #; cols = cols/np.max(dos_int) # color signaling each contribution
plt.plot(es,d2idv2,c="green",linewidth=3,label="Total")
plt.plot(es,ratio*cs_int,c="blue",linestyle='--',label="IETS Spin-flip")
plt.plot(es,dos_int,c="red",linestyle='--',label="Electronic")
plt.xlabel("Energy") ; plt.ylabel("simulated dIdV")
plt.xlim([min(es),max(es)]) ; plt.ylim([0.,np.max(d2idv2)*1.1]) ; plt.yticks([])
plt.legend()
plt.tight_layout()

plt.savefig("IETS.png",dpi=500)

plt.show()
