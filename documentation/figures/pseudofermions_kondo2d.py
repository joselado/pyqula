import numpy as np
import style
from pyqula import geometry
from pyqula.kondolattice import KondoLatticeHamiltonian

hc = geometry.square_lattice().get_hamiltonian(has_spin=True)
h = KondoLatticeHamiltonian(hc)
Jp = 2.0 # J' of the Kondo term J'(c^dag sigma c).S, in units of t
h2 = h.get_mean_field_hamiltonian(J=2*Jp, filling=0.5, nk=40,
        mf=([Jp+0.0j], [0.0])) # half filling, the Kondo insulator

# path Gamma, X, M, Gamma, in units of the reciprocal lattice vectors
corners = np.array([[0.,0.,0.], [0.5,0.,0.], [0.5,0.5,0.], [0.,0.,0.]])
def path(n): # n points on each of the three segments
    return np.concatenate([np.linspace(corners[i], corners[i+1], n,
            endpoint=False) for i in range(3)] + [corners[-1:]])
(ks, es, A) = h2.get_composite_spectral_function(kpath=path(100),
        energies=np.linspace(-6., 6., 300), delta=0.05)
ws = np.linspace(0., 5., 150)
chi = []
for q in path(20): # the susceptibility with the residual interaction
    (ws, c) = h2.get_moment_spinchi(q=q, energies=ws, nk=30, delta=0.05,
            U=-2.0)
    chi.append(c[:,0,0].imag)
chi = np.array(chi)

def draw(ax, c):
    im = ax[0].pcolormesh(np.arange(len(ks))/100, es, A["FF"].T, cmap=c.cmap,
                          vmin=0, shading="nearest", rasterized=True)
    ax[0].set_ylabel("energy [$t$]")
    ax[0].set_title("composite fermion, $A_{FF}$")
    im = ax[1].pcolormesh(np.arange(len(chi))/20, ws, chi.T, cmap=c.cmap,
                          vmin=0, vmax=3, shading="nearest", rasterized=True)
    ax[1].set_ylabel(r"frequency $\omega$ [$t$]")
    ax[1].set_title(r"Im $\chi(q,\omega)$, $U'=-2t$")
    for a in ax: # the corners of the path are at 0, 1, 2 and 3
        a.set_xticks([0, 1, 2, 3], [r"$\Gamma$", "X", "M", r"$\Gamma$"])
        for i in (1, 2): a.axvline(i, color=c.ink2, lw=0.6, alpha=0.5)

style.save(draw, "pseudofermions_kondo2d", npanels=2)
