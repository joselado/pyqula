import numpy as np
import style
from pyqula import geometry
from pyqula.spinon import SpinonHamiltonian

np.random.seed(1) # fix the random guess, so that the figure is reproducible
g = geometry.triangular_lattice() # a canonical frustrated-Heisenberg lattice
h = SpinonHamiltonian(g) # zero bare hopping, couplings come from J1/J2/...
h2 = h.get_mean_field_hamiltonian(J1=1.0,nk=12,mix=0.1,maxerror=1e-4)
(k,e) = h2.get_bands() # spinon dispersion

h = SpinonHamiltonian(g)
h.add_zeeman([0., 0., 0.3])            # or h.add_exchange([0.,0.,0.3])
h2 = h.get_mean_field_hamiltonian(J1=1.0, nk=12)
(kb,eb,sz) = h2.get_bands(operator="sz") # spinon dispersion in the field, and Sz
m = h2.get_magnetization()             # induced <S> per site

def draw(axs, c):
    axs[0].scatter(k, e, s=3, color=c.blue)
    sc = axs[1].scatter(kb, eb, c=sz, s=3, cmap=c.diverging, vmin=-1, vmax=1)
    style.colorbar(sc, axs[1], r"spin $\langle S_z\rangle$")
    axs[0].set_title("no field")
    axs[1].set_title(r"field $b_z=0.3$, $\langle\sigma_z\rangle=%.3f$" % m[0][2])
    for ax in axs:
        n = k.max() # the path goes Gamma-K-M-K'-Gamma, K at one third
        ax.set_xticks([0, n/3, n/2, 2*n/3, n], [r"$\Gamma$", "K", "M", "K'", r"$\Gamma$"])
        ax.set_xlim(0, n)
        ax.axhline(0, color=c.ink, lw=0.8, ls="--") # Fermi level
        ax.set_xlabel("momentum")
    axs[0].set_ylabel("spinon energy [$J$]")

style.save(draw, "pseudofermions_spinon", npanels=2)
print(e.min(), e.max(), eb.min(), eb.max(), m)
