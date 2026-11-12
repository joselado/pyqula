import numpy as np
import style
from pyqula import geometry
from pyqula.kondolattice import KondoLatticeHamiltonian

hc = geometry.chain().get_hamiltonian(has_spin=True) # conduction electrons
h = KondoLatticeHamiltonian(hc)
Jp = 2.0 # J' of the Kondo term J'(c^dag sigma c).S, in units of t
h2 = h.get_mean_field_hamiltonian(J=2*Jp, filling=0.5, nk=200,
        mf=([Jp+0.0j], [0.0])) # half filling, the Kondo insulator
energies = np.linspace(0., 6., 200)
qs = np.linspace(0., 1., 61) # momentum, in units of 2 pi
bare, rpa = [], []
for q in qs:
    (es, chi0) = h2.get_moment_spinchi(q=[q,0.,0.], energies=energies,
            nk=200, delta=0.05) # two f propagators
    (es, chi) = h2.get_moment_spinchi(q=[q,0.,0.], energies=energies,
            nk=200, delta=0.05, U=-2.0) # with the residual interaction
    bare.append(chi0[:,0,0].imag)
    rpa.append(chi[:,0,0].imag)
bare, rpa = np.array(bare), np.array(rpa)

def draw(ax, c):
    vmax = 2*bare.max()
    for (a, m, title) in [(ax[0], bare, "mean field"),
                          (ax[1], rpa, "RPA, $U'=-2t$")]:
        im = a.pcolormesh(2*np.pi*qs, es, m.T, cmap=c.cmap, vmin=0,
                          vmax=vmax, shading="nearest", rasterized=True)
        style.kpath(a, 2*np.pi*qs, ["0", r"$\pi$", r"$2\pi$"])
        a.set_xlabel("momentum $q$")
        a.set_title(title)
    ax[0].set_ylabel(r"frequency $\omega$ [$t$]")
    style.colorbar(im, ax[1], r"Im $\chi(q,\omega)$")

style.save(draw, "pseudofermions_momentchi", npanels=2)
