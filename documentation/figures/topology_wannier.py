import numpy as np
import style
from pyqula import geometry
from pyqula import topology
g = geometry.honeycomb_lattice() # create honeycomb lattice
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_haldane(0.05) # Add Haldane coupling
C = h.get_chern(integration="wannier") # winding of the hybrid Wannier centers
w = topology.z2_wannier_centers(h,full=True) # flow of the centers over the whole BZ
k2 = w[0] # momentum along the pumping direction
x = np.array(w[1:])/(2*np.pi) # centers, in units of the lattice constant

def draw(ax, c):
    for xi in x: # one row per occupied band, the two spins on top of each other
        for shift in [-1,0,1]: # periodic copies, so that the winding is visible
            ax.scatter(k2, xi+shift, s=10, color=c.blue)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"momentum $k_2$ [$|\mathbf{b}_2|$]")
    ax.set_ylabel(r"Wannier center $x_1$ [$a$]")
    ax.set_title(f"Haldane model, C = {C:.0f}")

style.save(draw, "topology_wannier")
