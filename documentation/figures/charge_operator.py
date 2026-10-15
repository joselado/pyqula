import style
from pyqula import geometry
import numpy as np
g = geometry.chain()
h = g.get_hamiltonian(has_spin=True)
h.add_zeeman([0.,0.,0.5]) # Zeeman field along z, splits the bands by 1.0
sx = h.get_operator("sx") # any of the operators from the "Operators" section
es = np.linspace(0.,2.,401)
(es,chi) = h.get_chi(A=sx,B=sx,q=[0.02,0.,0.],energies=es,nk=2001,delta=0.02)
(es,chic) = h.get_chi(q=[0.02,0.,0.],energies=es,nk=2001,delta=0.02) # charge, as a reference

def draw(ax, c):
    ax.plot(es, chi.imag, color=c.blue, label=r"spin flip, $A=B=S_x$")
    ax.plot(es, chic.imag, color=c.orange, label="charge")
    ax.axvline(1.0, color=c.ink2, lw=0.8, ls="--")
    ax.text(1.02, 0.95*chi.imag.max(), "Zeeman\nsplitting", color=c.ink2,
            va="top")
    ax.set_xlabel(r"frequency $\omega$ [$t$]")
    ax.set_ylabel(r"Im $\chi(q,\omega)$, $q=0.02$")
    ax.legend(loc="upper left")

style.save(draw, "charge_operator")
