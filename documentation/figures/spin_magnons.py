from pyqula import geometry
import numpy as np
import style
g = geometry.bichain() # two sites per cell, so Neel order fits in the cell
h = g.get_hamiltonian()
seed = h.copy() ; seed.add_antiferromagnetism(0.5) # symmetry-breaking seed
hmf = h.get_mean_field_hamiltonian(U=3.0,nk=100,mf=seed,filling=0.5)
qs,ws,gammas = hmf.get_magnon_bands(nq=40,energies=np.linspace(0.01,3.,200),delta=2e-2,nk=40)
sharp = np.abs(gammas) < 0.05 # sharp modes, the others are damped

def draw(ax, c):
    ax.scatter(qs[~sharp], ws[~sharp], s=6, color=c.ink2, alpha=0.4,
               label="damped")
    ax.scatter(qs[sharp], ws[sharp], s=14, color=c.blue, label="sharp")
    style.kpath(ax, qs, ["0", r"$\pi$", r"$2\pi$"])
    ax.set_xlabel("momentum $q$ along the chain")
    ax.set_ylabel(r"frequency $\omega$ [$t$]")
    ax.set_ylim(0, 3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)

style.save(draw, "spin_magnons")
