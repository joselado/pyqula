import numpy as np
import style
from pyqula import geometry
g = geometry.honeycomb_lattice() # create a honeycomb lattice
h = g.get_hamiltonian(has_spin=False) # get the Hamiltonian
h.add_haldane(0.2) # Chern insulator
h.shift_fermi(0.3) # put the Fermi energy in the gap

(ws,s) = h.get_optical_conductivity(energies=np.linspace(0.,4.,100),
                                    nk=40,T=0.02,delta=0.05)
absorption = s[:,0,0].real # Re sigma_xx, the optical absorption

# the DC Hall response is quantized to the Chern number
(w0,s0) = h.get_optical_conductivity(energies=[0.],nk=40,T=0.01,delta=1e-3)
print("sigma_xy(0) in e^2/h:",2.*np.pi*s0[0,0,1].real) # -1

gap = h.get_gap() # the band gap, to mark the absorption edge

def draw(ax, c):
    ax.plot(ws, 2*np.pi*absorption, color=c.blue,
            label=r"Re $\sigma_{xx}$")
    ax.plot(ws, 2*np.pi*s[:,0,1].real, color=c.orange,
            label=r"Re $\sigma_{xy}$")
    ax.axvline(gap, color=c.ink2, lw=1.0, ls="--")
    ax.text(gap-0.05, 0.97, "gap", color=c.ink2, va="top", ha="right",
            transform=ax.get_xaxis_transform())
    ax.axhline(0, color=c.ink, lw=0.6, alpha=0.4)
    ax.set_xlabel(r"frequency $\hbar\omega$ [$t$]")
    ax.set_ylabel(r"conductivity [$e^2/h$]")
    ax.legend(loc="upper right")

style.save(draw, "optical_chern")
