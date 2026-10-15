import numpy as np
import style
from pyqula import geometry

# triangular lattice
g = geometry.triangular_lattice() # get the geometry
h = g.get_hamiltonian()  # get the Hamiltonian
(es,ds) = h.get_dos(nk=400,delta=0.05) # the default broadening, 5/nk at nk=100, on a finer k-mesh

# ED and Green's function on a chain with Rashba spin-orbit coupling
g = geometry.chain()
h = g.get_hamiltonian(tij=[0.5,0.,0.,0.5],has_spin=True)
h.add_rashba(0.7)
energies = np.linspace(-4.,4.0,60)
(e1,d1) = h.get_dos(energies=energies,delta=1e-2,mode="ED",nk=1000)
(e2,d2) = h.get_dos(energies=energies,delta=1e-2,mode="Green")

# sublattice-projected DOS of the honeycomb lattice
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=True)
h.add_sublattice_imbalance(.4)
(es3,ds3) = h.get_dos(operator="sublattice",nk=200,delta=5e-2)

def draw(ax, c):
    ax[0].fill_between(es, ds, color=c.blue, alpha=0.3, lw=0)
    ax[0].plot(es, ds, color=c.blue)
    ax[0].set_title("triangular lattice")
    ax[1].plot(e1, d1, color=c.blue, label="ED")
    ax[1].plot(e2, d2, "o", color=c.orange, ms=4, label="Green")
    ax[1].set_title("chain with Rashba")
    ax[1].legend(loc="upper right")
    ax[2].fill_between(es3, ds3, where=ds3>0, color=c.blue, alpha=0.5, lw=0)
    ax[2].fill_between(es3, ds3, where=ds3<0, color=c.orange, alpha=0.5, lw=0)
    ax[2].plot(es3, ds3, color=c.ink, lw=1)
    ax[2].axhline(0, color=c.ink, lw=0.6, alpha=0.4)
    ax[2].set_title("honeycomb, sublattice A minus B")
    for a in ax: a.set_xlabel("energy [$t$]")
    ax[0].set_ylabel(r"DOS $D(\omega)$ [states/$t$]")
    ax[2].set_ylabel(r"$D_A(\omega)$ [states/$t$]")

style.save(draw, "bands_dos", npanels=3)
