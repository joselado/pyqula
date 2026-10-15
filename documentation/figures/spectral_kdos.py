import numpy as np
import style
from pyqula import geometry

g = geometry.triangular_lattice() # get the geometry
h = g.get_hamiltonian()  # get the Hamiltonian
(k,e,d) = h.get_kdos_bands(energies=np.linspace(-4,7,1101),nk=600,delta=0.05) # the whole band, on a dense path

nk, ne = len(np.unique(k)), len(np.unique(e)) # k is the outer index
(k,e,d) = [a.reshape(nk,ne) for a in (k,e,d)]

def draw(ax, c):
    im = ax.pcolormesh(k, e, d, cmap=c.cmap, shading="auto", vmax=np.percentile(d,99))
    style.colorbar(im, ax, r"spectral function $A(k,\omega)$")
    style.kpath(ax, k, [r"$\Gamma$", "K", "M", "K'", r"$\Gamma$"], at=[0,1/3,1/2,2/3,1])
    ax.set_xlabel("momentum along the k-path")
    ax.set_ylabel(r"energy $\omega$ [$t$]")

style.save(draw, "spectral_kdos")
