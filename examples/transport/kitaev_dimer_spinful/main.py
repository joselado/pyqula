# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import geometry
from pyqula import heterostructures
import numpy as np
import matplotlib.pyplot as plt
g = geometry.chain()
h = g.get_hamiltonian()
hl = h.copy() # copy
hl.add_swave(.0) # add electron hole symmetry
hl.add_onsite(0.8)
hd = h.copy() # copy

# create a topological superconductor
full = True
if full: # full spinful model
    hd.add_rashba(0.5) # Rashba SOC
    hd.add_zeeman(2.0) # Zeeman
    hd.add_onsite(2.0) # net onsite
    hd.add_swave(.5) # pairing gap of 0.1
else: # minimal kitaev superconductor model
    hd.add_onsite(5) # make a large shift of the chemical potential
    hd.add_zeeman([0.,0.,5]) # put a single band at the chemical potential
    Dt = 0.9 # this is the ratio Delta/t of your paper
    hd.add_pairing(mode="pwave",delta=Dt,d=[1.0,0.,0.])



def get(mu1,mu2):
    hd1 = hd.copy() ; hd1.add_onsite(mu1)
    hd2 = hd.copy() ; hd2.add_onsite(mu2)
    
    hc = [hd1,hd2] # dimer 
    ht = heterostructures.build(hl,hl,central=hc) # create the junction
    ht.scale_lc = 0.4 # scale the left coupling
    ht.scale_rc = 0.4 # scale the left coupling
    ht.delta = 1e-3 # imaginary part of the energy
    V = 0. # bias
    G = ht.didv(energy=V) # conductance
    return G

V = 4.0 # range of potentials
mus1 = np.linspace(-V,V,40)
mus2 = np.linspace(-V,V,40)

Gs = [[get(mu1,mu2) for mu1 in mus1] for mu2 in mus2]

fig = plt.figure(figsize=(6,5))
plt.contourf(mus1,mus2,Gs,levels=100,cmap="Blues")
#plt.imshow(Gs)
plt.colorbar(ticks=[0.,np.max(Gs)],label="Conductance")
plt.axis("equal") ; plt.xlabel("$\\mu_1$") ; plt.ylabel("$\\mu_2$")
plt.show()
