import numpy as np
import style
from pyqula import geometry
from pyqula.specialhamiltonian import NbSe2

g = geometry.triangular_lattice() # get the geometry
h = g.get_hamiltonian() # get the Hamiltonian
(kx,ky,fs) = h.get_fermi_surface(e=0.0,nk=150,delta=1e-1)

h = NbSe2(soc=0.9) # multi-orbital spin-orbit-coupled Hamiltonian
(kx2,ky2,fs2) = h.get_fermi_surface(e=0.,nk=100,delta=3e-1,operator="sz")

def draw(ax, c):
    im = ax[0].tripcolor(kx, ky, fs, cmap=c.cmap, shading="gouraud")
    style.colorbar(im, ax[0], r"spectral weight $A(\mathbf{k},0)$")
    ax[0].set_title("triangular lattice")
    m = np.abs(fs2).max()
    im = ax[1].tripcolor(kx2, ky2, fs2, cmap=c.diverging, vmin=-m, vmax=m, shading="gouraud")
    style.colorbar(im, ax[1], r"spin-resolved weight, $\langle s_z\rangle$")
    ax[1].set_title(r"NbSe$_2$ model")
    for a in ax:
        a.set_aspect("equal")
        a.set_xlabel(r"$k_x$ [$2\pi/a$]")
        a.set_ylabel(r"$k_y$ [$2\pi/a$]")

style.save(draw, "spectral_fermi", npanels=2)
