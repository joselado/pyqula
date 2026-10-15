import style
from pyqula import geometry
import numpy as np
g = geometry.triangular_lattice() # geometry of a triangular lattice
h = g.get_hamiltonian()  # get the Hamiltonian
h.setup_nambu_spinor() # setup the Nambu form of the Hamiltonian
h = h.get_mean_field_hamiltonian(U=-2.0,filling=
                   0.45,mf="swave",nk=40) # perform SCF
# electron spectral-function
(k,w,A) = h.get_kdos_bands(operator="electron",nk=400,
                   energies=np.linspace(-2.0,2.0,200),delta=0.03)
k, w, A = k.reshape(400,200), w.reshape(400,200), A.reshape(400,200)

def draw(ax, c):
    pm = ax.pcolormesh(k, w, A, cmap=c.cmap, vmax=0.5*A.max(), shading="auto")
    style.colorbar(pm, ax, "electron spectral function")
    ax.set_xticks([0, 1/3, 1/2, 2/3, 1], [r"$\Gamma$", "K", "M", "K'", r"$\Gamma$"])
    ax.set_xlabel("momentum")
    ax.set_ylabel(r"energy $\omega$ [$t$]")

style.save(draw, "meanfield_swave")
print("Delta", h.extract("swave"))
(kb,eb) = h.get_bands(); print("gap", 2*abs(eb).min())
