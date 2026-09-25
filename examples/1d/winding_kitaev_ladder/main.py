# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Winding number and Zak phase of a Kitaev chain, and of two Kitaev chains
# coupled by a weak rung hopping, across the band: the winding counts the
# Majorana zero modes at each end (1 and 2 inside the band), while the Zak
# phase only knows their parity (pi and 0)
import numpy as np
from pyqula import geometry
from pyqula import topology

def xpwave(r1,r2): # p-wave pairing on the bonds along x only
    dr = r1-r2
    if abs(np.linalg.norm(dr)-1.)<1e-4 and abs(dr[1])<1e-6:
        return dr[0]*np.array([[0.,1.],[1.,0.]],dtype=complex)
    return np.zeros((2,2),dtype=complex)

def kitaev(g,mu,tperp=0.1):
    def fun(r1,r2): # hopping 1 along the chains, tperp on the rungs
        dr = r1-r2
        if abs(np.linalg.norm(dr)-1.)>1e-4: return 0.
        return 1. if abs(dr[1])<1e-6 else tperp
    h = g.get_hamiltonian(fun=fun) # spinful Hamiltonian
    h.add_onsite(20.+mu) # shift both spin bands up
    h.add_zeeman([0.,0.,20.]) # and bring one of them back
    h.add_pairing(mode=xpwave,delta=0.3) # p-wave pairing
    return h

for (name,g) in [("chain",geometry.chain()),("ladder",geometry.ladder())]:
    for mu in [-3.,-1.,0.,1.,3.]:
        h = kitaev(g,mu)
        W = h.get_winding_number() # winding number
        zak = topology.berry_phase(h,nk=100,write=False)/np.pi # Zak phase/pi
        print(name,"mu =",mu,"  W =",W,"  Zak phase/pi =",round(abs(zak),3))
