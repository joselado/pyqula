import numpy as np
import style
from pyqula import geometry
from pyqula import kdos
g = geometry.honeycomb_lattice() # create honeycomb lattice
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_soc(0.05) # Add spin-orbit coupling
from pyqula import topology
z2 = topology.z2_invariant(h) # Z2 invariant
h0 = g.get_hamiltonian() # the trivial insulator, for comparison
h0.add_sublattice_imbalance(0.2) # gapped by a sublattice imbalance
z0 = topology.z2_invariant(h0) # Z2 invariant
w = topology.z2_wannier_centers(h) # flow of the centers over half of the BZ
w0 = topology.z2_wannier_centers(h0)

def draw(ax, c):
    for (axi,wi,zi,name) in zip(ax,[w,w0],[z2,z0],["Kane-Mele","sublattice imbalance"]):
        for xi in np.array(wi[1:])/(2*np.pi): # one row per occupied band
            axi.scatter(wi[0], xi, s=10, color=c.blue)
        axi.set_ylim(-0.5, 0.5)
        axi.set_xlim(0, 0.5)
        axi.set_xlabel(r"momentum $k_2$ [$|\mathbf{b}_2|$]")
        axi.set_title(f"{name}, $Z_2$ = {zi:+.0f}")
    ax[0].set_ylabel(r"Wannier center $x_1$ [$a$]")

style.save(draw, "topology_z2", npanels=2)
