import numpy as np
import style
from pyqula import geometry
g = geometry.chain()
g = g.get_supercell(3000) # a big supercell
g.dimensionality = 0
h = g.get_hamiltonian(is_sparse=True,has_spin=False)

from pyqula import kpm
(x,y) = kpm.dm_ij_energy(h.intra,npol=200,i=0,j=9,ne=1000)

js = range(1500,1531) # sites j, from the middle of the chain
gij = np.array([kpm.dm_ij_energy(h.intra,npol=200,i=1500,j=j,ne=1000)[1].real for j in js])

def draw(ax, c):
    vm = np.abs(gij).max()
    m = ax.pcolormesh(np.array(js)-1500, x, gij.T, cmap=c.diverging, vmin=-vm, vmax=vm,
                      shading="nearest")
    style.colorbar(m, ax, r"$\pi\langle i|\delta(E-H)|j\rangle$")
    ax.set_xlabel("distance $j-i$ [sites]")
    ax.set_ylabel("energy [$t$]")
    ax.set_ylim(-2.5, 2.5)

style.save(draw, "kpm_green")
