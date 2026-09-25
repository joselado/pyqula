# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Spin Chern number (occupied states split by the sign of P sz P) of the
# Kane-Mele model as a Rashba coupling grows, against the Z2 invariant and
# the sz-weighted integral of topology.spin_chern; and the mirror Chern
# number as an out-of-plane exchange field breaks time reversal
import numpy as np
from pyqula import geometry
from pyqula import topology

g = geometry.honeycomb_lattice()
for r in [0.,0.05,0.1,0.15]: # Rashba coupling
    h = g.get_hamiltonian() # create hamiltonian of the system
    h.add_kane_mele(0.1) # Add spin-orbit coupling
    h.add_rashba(r) # Add Rashba coupling
    cs = h.get_spin_chern(nk=20) # spin Chern number
    z2 = topology.z2_invariant(h,nk=30,nt=30) # Z2 invariant
    w = topology.spin_chern(h,nk=20) # sz-weighted Berry curvature integral
    print("Rashba =",r,"  C_s =",round(cs,3),"  Z2 =",z2,"  sz-weighted =",round(w,3))
for m in [0.,0.2,0.4]: # out-of-plane exchange field
    h = g.get_hamiltonian() # create hamiltonian of the system
    h.add_kane_mele(0.1) # Add spin-orbit coupling
    h.add_exchange([0.,0.,m]) # breaks time reversal, keeps the mirror
    cm = h.get_mirror_chern(nk=20) # mirror Chern number
    c = h.get_chern(nk=20) # Chern number
    print("exchange =",m,"  C_M =",round(cm,3),"  C =",round(c,3))
