# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Spinon mean field of a Heisenberg model in a large unit cell with
# inequivalent sites, its stability, and its response to a Zeeman field.
#
# The cell has 36 sites of the triangular lattice. The exchange has first
# neighbors J1 and second neighbors J2 = 0.11 J1, and both change by 20
# percent across the cell, along its two vectors. The modulation is written
# with the fractional coordinates of the bond, so it has the period of the
# cell, and a bond has the same exchange seen from its two ends, as a spin
# model needs. Its origin is not a symmetry point of the lattice, so the 36
# sites are all different. We keep only the bond channel of the exchange
# (decoupling="bond") and take the pi-flux pattern as ansatz, with flux pi
# in the up triangles and zero in the down ones (a Dirac spin liquid).
#
# With hold="phases" the amplitude of each hopping is free, and with
# inequivalent bonds the loop can go to a dimerized state. With
# hold="pattern" the hoppings stay t_ij = c J_ij P_ij with a single
# amplitude c, and each site keeps its own Lagrange multiplier, so that
# there is one fermion on every site.
#
# The stiffness, the smallest eigenvalue of 1 + J(q) chi0(q,0), says if the
# liquid is stable against magnetic order. With a Zeeman field the loop adds
# the Weiss field of the neighboring moments, sum_j J_ij <S_j>. The uniform
# transverse spin response then has its peak at the Zeeman energy, whatever
# the exchange is, because the total spin commutes with the exchange. If
# the stiffness were negative, the field would drive the loop to a magnetic
# order instead.

import numpy as np
from pyqula import geometry
from pyqula.spinon import SpinonHamiltonian

g0 = geometry.triangular_lattice()
g = g0.get_supercell((6,6)) # 36 sites
n = len(g.r)
# origin of the modulation, away from the symmetry points of the lattice
origin = np.mean(g.r,axis=0) + g0.a1/6. + g0.a2/3.
cell = np.linalg.inv(np.array([g.a1[:2],g.a2[:2]]).T) # to fractional coordinates
def J(r1,r2):
    """Exchange of first and second neighbors, modulated with the period
    of the cell along its two vectors"""
    d = np.linalg.norm(r1-r2)
    u = cell@((r1+r2)/2. - origin)[:2] # fractional coordinates of the bond
    mod = 1.0 + 0.1*(np.cos(2*np.pi*u[0]) + np.cos(2*np.pi*u[1]))
    if abs(d-1.)<1e-3: return mod # first neighbors
    if abs(d-np.sqrt(3.))<1e-3: return 0.11*mod # second neighbors
    return 0.

# pi-flux pattern: a bond along a1 is +1, and any other first-neighbor bond
# is (-1)^p, with p the coordinate along a1 of its lower end
A = np.array([g0.a1[:2],g0.a2[:2]]).T
ansatz = dict()
for (d,m) in g.get_hamiltonian(has_spin=True).get_dict().items():
    m = np.array(m)
    for i in range(n):
        for j in range(n):
            if abs(m[2*i,2*j])<1e-9: continue # not a bond
            dr = g.r[j] + d[0]*g.a1 + d[1]*g.a2 - g.r[i] # the bond
            if abs(dr[1])<1e-6: continue # a bond along a1 keeps +1
            low = g.r[i] if dr[1]>0 else g.r[i] + dr # its lower end
            p = int(round(np.linalg.solve(A,(low-g.r[0])[:2])[0]))
            m[2*i:2*i+2,2*j:2*j+2] *= (-1)**p
    ansatz[d] = m

def hopping_range(h):
    """Smallest and largest |t_ij|/J_ij over the first-neighbor bonds"""
    out = []
    for (d,m) in h.get_multihopping().get_dict().items():
        m = np.array(m)[0::2,0::2]
        for i in range(n):
            for j in range(n):
                rj = g.r[j] + d[0]*g.a1 + d[1]*g.a2
                if abs(np.linalg.norm(rj-g.r[i])-1.)<1e-3:
                    out.append(abs(m[i,j])/J(g.r[i],rj))
    return min(out),max(out)

for hold in ["phases","pattern"]:
    h = SpinonHamiltonian(g) # no bare hopping, a pure spin model
    h2 = h.get_mean_field_hamiltonian(Jr=J,rcut=2.0,decoupling="bond",
            ansatz=ansatz,hold=hold,nk=3,T=0.02,mix=0.5)
    print("hold =",hold)
    print("  |t_ij|/J_ij between %.3f and %.3f" % hopping_range(h2))
    print("  largest error in the occupation",
          np.max(np.abs(h2.local_occupation-1.)))
    print("  spread of the Lagrange multipliers",
          np.ptp(h2.constraint_lambda))

# stability of the state with the held pattern, at a few momenta of the cell
print("Stiffness of the held pattern")
for q in [[0.,0.,0.],[1/3,1/3,0.],[1/2,0.,0.],[0.,1/2,0.],[1/2,1/2,0.]]:
    print("  q =",np.round(q,3),h2.get_rpa_stiffness(q,nk=3,delta=1e-3,T=0.02))

# the same cell in a Zeeman field, with the Weiss field in the loop
b = 0.1 # the field multiplies sigma_z, so the Zeeman energy is 2b
h = SpinonHamiltonian(g)
h.add_zeeman([0.,0.,b])
h2 = h.get_mean_field_hamiltonian(Jr=J,rcut=2.0,decoupling="bond",
        ansatz=ansatz,hold="pattern",nk=3,T=0.02,mix=0.5)
m = h2.get_magnetization()[:,2] # induced <sigma_z> of each site
print("Induced moment between %.4f and %.4f" % (m.min(),m.max()))
energies = np.linspace(0.,0.6,61)
# the response takes the k-mesh and the temperature of the loop
(es,chi) = h2.get_spinchi_full(q=[0.,0.,0.],energies=energies,delta=0.01,
        nk=3,T=0.02)
sx = np.array([np.sum(c[0:n,0:n]).imag for c in chi]) # uniform Sx response
print("Peak of the uniform transverse response at energy",es[np.argmax(np.abs(sx))])
print("Zeeman energy",2*b)
