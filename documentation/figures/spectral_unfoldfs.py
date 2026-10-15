import numpy as np
import style
from pyqula import geometry

# point defect in a 3 x 3 supercell
g0 = geometry.triangular_lattice()
n = 3 # size of the supercell
g = g0.get_supercell(n,store_primal=True)
h = g.get_hamiltonian()
h.add_onsite(lambda r: 100.0 if np.linalg.norm(r-g.r[0])<1e-1 else 0.0) # a point defect

out = h.get_multi_fermi_surface(nk=50,energies=np.linspace(-4,4,100),delta=0.1,
        nsuper=n,operator="unfold")

# charge order in a sqrt(3) x sqrt(3) supercell
g0 = geometry.triangular_lattice() # primitive geometry
g = g0.get_supercell(np.sqrt(3),store_primal=True) # sqrt(3) x sqrt(3) supercell
h = g.get_hamiltonian() # Hamiltonian of the supercell
h.add_onsite(lambda r: 0.6 if np.linalg.norm(r-g.r[0])<1e-1 else 0.0) # charge order
(kx,ky,fs) = h.get_fermi_surface(e=0.0,nk=100,delta=0.1,operator="unfold",
        primal_mesh=True) # unfolded Fermi surface, on the primitive mesh

(kx1,ky1,fs1) = out[49] # one row per energy, this one at omega=-0.04
kx1, ky1 = kx1/n, ky1/n # in units of the primitive cell, as the second map

def draw(ax, c):
    im = ax[0].tripcolor(kx1, ky1, fs1, cmap=c.cmap, shading="gouraud")
    style.colorbar(im, ax[0], "unfolded weight")
    ax[0].set_title(r"point defect, $3\times 3$")
    im = ax[1].tripcolor(kx, ky, fs, cmap=c.cmap, shading="gouraud")
    style.colorbar(im, ax[1], "unfolded weight")
    ax[1].set_title(r"charge order, $\sqrt{3}\times\sqrt{3}$")
    for a in ax:
        a.set_aspect("equal")
        a.set_xlabel(r"$k_x$ [$2\pi/a$]")
        a.set_ylabel(r"$k_y$ [$2\pi/a$]")

style.save(draw, "spectral_unfoldfs", npanels=2)
