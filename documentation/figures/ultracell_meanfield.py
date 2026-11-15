import numpy as np
import style
from pyqula import geometry

g = geometry.honeycomb_lattice() # unit cell of the honeycomb lattice
h = g.get_hamiltonian() # spinful, first-neighbor hopping t=1
h.add_sublattice_imbalance(1.0) # +1 on A and -1 on B, a gap of 2t
n = 24 # period of the modulation, in unit cells along a1
hs = h.get_supercell([n,1]) # the ultracell, written as a supercell
b1 = np.linalg.inv(np.array([g.a1,g.a2,g.a3]).T)[0] # b1.r is the coordinate along a1
B0 = 0.05 # amplitude of the exchange field, in units of t
hs.add_exchange(lambda r: [0.,0.,B0*np.cos(2*np.pi*b1.dot(r)/n)]) # along z
u = h.get_ultracell(hs) # hs solved in the Bloch states of the unit cell
nk = [2,12] # k-mesh of the ultracell: the ultracell is 24 cells long along a1
m0 = u.get_vev(operator="sz",nk=nk) # induced moment without interaction
u2 = u.get_mean_field_hamiltonian(U=1.5,filling=0.5,mf="ferroZ",nk=nk) # Hubbard U
m = u2.get_vev(operator="sz",nk=nk) # induced moment with the Hubbard U
x = np.array([b1.dot(r) for r in u2.hs.geometry.r]) # coordinate along a1

o = np.argsort(x)
print("largest moment, U=0:",np.max(np.abs(m0)),", U=1.5:",np.max(np.abs(m)),
      ", ratio:",np.max(np.abs(m))/np.max(np.abs(m0)))
print("ratio on A and on B:",[np.max(np.abs(m[s]))/np.max(np.abs(m0[s]))
      for s in (u2.hs.geometry.sublattice>0,u2.hs.geometry.sublattice<0)])
print("largest in-plane moment:",np.max(np.abs(u2.get_vev(operator="sx",nk=nk))))

def draw(ax, c):
    ax.plot(x[o], 1e3*m0[o], ".", ms=4, color=c.ink2, label=r"$U=0$")
    ax.plot(x[o], 1e3*m[o], ".", ms=4, color=c.blue, label=r"$U=1.5t$")
    ax.set_xlabel(r"position along $\mathbf{a}_1$ [unit cells]")
    ax.set_ylabel(r"induced moment $\langle S_z\rangle$ [$10^{-3}$]")
    ax.legend(frameon=False)

style.save(draw, "ultracell_meanfield")
