import numpy as np
import style
from pyqula import geometry
np.random.seed(0) # fix the random guess, so that the figure is reproducible
g = geometry.triangular_lattice() # generate the geometry
h = g.get_hamiltonian() # create Hamiltonian of the system
h.add_exchange([0.,0.,1.]) # add exchange field
h.setup_nambu_spinor() # initialize the Nambu basis
# perform a superconducting non-collinear mean-field calculation
h = h.get_mean_field_hamiltonian(V1=-1.5,
                     filling=0.3,mf="random",nk=20) # nk=20 here, to keep it short
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

style.save(draw, "meanfield_triplet")
(kb,eb) = h.get_bands(); print("gap", 2*abs(eb).min())
print(h.get_magnetization())
for n in ["swave"]: print(n, h.extract(n))
