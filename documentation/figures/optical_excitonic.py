import numpy as np
import style
from pyqula import geometry
from pyqula.bsetk.interaction import density_interaction
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False) # spinless, one spin species
h.add_sublattice_imbalance(1.0) # gapped honeycomb lattice, gap 2t at K
W = density_interaction(h,Vr=lambda r1,r2: 0.8/np.sqrt((r1-r2).dot(r1-r2)+0.25)) # Coulomb tail
zero = {(0,0,0):np.zeros(h.intra.shape,dtype=np.complex128)} # no interaction

kw = dict(energies=np.linspace(0.,2.6,261),nk=30,delta=0.04)
(ws,s) = h.get_excitonic_optical_conductivity(V=W,component="xx",**kw)
(ws,s0) = h.get_excitonic_optical_conductivity(V=zero,kernel="none",component="xx",**kw)
(ws,chi) = h.get_excitonic_shg_susceptibility(V=W,components="xxx",**kw)
(ws,chi0) = h.get_excitonic_shg_susceptibility(V=zero,kernel="none",components="xxx",**kw)

def draw(ax, c):
    ax[0].plot(ws, 2*np.pi*s.real, color=c.blue, label="excitons")
    ax[0].plot(ws, 2*np.pi*s0.real, color=c.orange, label="no interaction")
    ax[0].set_xlabel(r"frequency $\hbar\omega$ [$t$]")
    ax[0].set_ylabel(r"Re $\sigma_{xx}$ [$e^2/h$]")
    ax[0].legend(loc="upper left")
    ax[1].plot(ws, np.abs(chi), color=c.blue, label="excitons")
    ax[1].plot(ws, np.abs(chi0), color=c.orange, label="no interaction")
    ax[1].set_xlabel(r"frequency $\hbar\omega$ [$t$]")
    ax[1].set_ylabel(r"$|\chi_{xxx}|$ [nm$^2$/V]")
    ax[1].legend(loc="upper left")

style.save(draw, "optical_excitonic", npanels=2)
