import style
from pyqula import geometry
g = geometry.honeycomb_lattice() # create a honeycomb lattice
h = g.get_hamiltonian(has_spin=False) # get the Hamiltonian
h.add_haldane(0.1) # Chern insulator, C = 1
print(h.get_chern(nk=20)) # 1.0

# Li-Haldane entanglement spectrum xi_n(k_par) across the BZ
(ks,xis) = h.get_entanglement_spectrum(nsuper=10,nk=101)

# entanglement entropy, per parallel unit cell, averaged over the BZ
print(h.get_entanglement_entropy(nsuper=10,nk=20))

h0 = g.get_hamiltonian(has_spin=False) # the trivial insulator, for comparison
h0.add_sublattice_imbalance(0.3) # gapped by a sublattice imbalance
(ks0,xis0) = h0.get_entanglement_spectrum(nsuper=10,nk=101)

def draw(ax, c):
    for (axi,k,xi,name) in zip(ax,[ks,ks0],[xis,xis0],["Haldane, C = 1","sublattice imbalance, C = 0"]):
        for col in xi.T: # one level per orbital of the region
            axi.scatter(k, col, s=8, color=c.blue)
        axi.axhline(0, color=c.ink, lw=0.6, alpha=0.4)
        axi.set_ylim(-8, 8)
        axi.set_xlabel(r"momentum $k_\parallel$ [$2\pi/a$]")
        axi.set_title(name)
    ax[0].set_ylabel(r"entanglement level $\xi_n$")

style.save(draw, "entanglement_spectrum", npanels=2)
