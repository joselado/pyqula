import numpy as np
import style
from pyqula import geometry

n = 20
g = geometry.chain().get_supercell(n)
h = g.get_hamiltonian(has_spin=False,non_hermitian=True,tij=[-1.])
# a purely imaginary Aubry-Andre modulation, with no Hermitian part at
# all: gain where the cosine is positive, loss where it is negative
h.add_onsite(lambda r: 0.6j*np.cos(2.*np.pi*r[0]/n))

(ks,es) = h.get_bands(kpath=[[0.,0.,0.]],write=False)
print("eigenvalues are complex:",es.dtype)
print("largest gain rate Im(E):",np.round(np.max(es.imag),4))

def draw(ax, c):
    ax.axhline(0, color=c.ink, lw=0.6, alpha=0.4)
    ax.scatter(es.real, es.imag, s=40, color=c.blue)
    ax.set_xlabel(r"energy Re$\,E$ [$t$]")
    ax.set_ylabel(r"gain rate Im$\,E$ [$t$]")

style.save(draw, "nonhermitian_spectrum")
