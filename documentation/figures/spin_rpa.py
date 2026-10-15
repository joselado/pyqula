from pyqula import geometry
import numpy as np
import style
from matplotlib.colors import LogNorm
g = geometry.bichain() # two sites per cell, so Neel order fits in the cell
h = g.get_hamiltonian(has_spin=True)
seed = h.copy() ; seed.add_antiferromagnetism(0.5) # symmetry-breaking seed
hmf = h.get_mean_field_hamiltonian(U=3.0,filling=0.5,mf=seed,nk=100) # magnetic state
(es,chis) = hmf.get_spinchi_ladder(energies=np.linspace(0.,2.,400),q=[0.1,0.,0.],nk=40,delta=2e-2)
(es,chi0) = hmf.get_spinchi_ladder(energies=np.linspace(0.,2.,400),q=[0.1,0.,0.],nk=40,delta=2e-2,
                                   RPA=False) # the bare response, as a reference

g2 = geometry.honeycomb_lattice() # already two sublattices per cell
h2 = g2.get_hamiltonian(has_spin=True)
hmf2 = h2.get_mean_field_hamiltonian(U=3.0,filling=0.5,mf="antiferro")
qdisp = hmf2.get_qdos_iets(energies=np.linspace(0.,2.,250),qpath=["G","K","M"],nq=120,nk=60,delta=1e-2)
(qs,ws,chimap) = qdisp # q-points, frequencies, Im Tr chi(q,w)

def draw(ax, c):
    im_rpa = np.trace(chis, axis1=1, axis2=2).imag # sum over the two sites
    im_0 = np.trace(chi0, axis1=1, axis2=2).imag
    ax[0].semilogy(es, im_rpa, color=c.blue, label="RPA")
    ax[0].semilogy(es, im_0, color=c.orange, label="bare")
    ax[0].set_ylim(1e-2, 2*im_rpa.max())
    ax[0].set_xlabel(r"frequency $\omega$ [$t$]")
    ax[0].set_ylabel(r"Im $\chi_{+-}(q,\omega)$")
    ax[0].set_title("chain, $q=0.1$")
    ax[0].legend(loc="upper right")
    x = np.arange(len(chimap))
    im = ax[1].pcolormesh(x, ws, np.clip(chimap.T, 1e-1, None), cmap=c.cmap,
                          norm=LogNorm(vmin=1e-1), shading="nearest",
                          rasterized=True)
    style.colorbar(im, ax[1], r"Im Tr $\chi(q,\omega)$")
    iK = np.argmin(np.abs(qs[:,0,1]-2/3)) # K = (-1/3,2/3) on this path
    ax[1].set_xticks([0, iK, x[-1]], [r"$\Gamma$", "K", "M"])
    ax[1].axvline(iK, color=c.ink2, lw=0.6, alpha=0.5)
    ax[1].set_ylabel(r"frequency $\omega$ [$t$]")
    ax[1].set_title("honeycomb")

style.save(draw, "spin_rpa", npanels=2)
