import numpy as np
import style
from pyqula import geometry
g = geometry.square_lattice() # geometry of the 2D model
h = g.get_hamiltonian() # generate the Hamiltonian
h.add_onsite(-0.6) # move away from half filling
h.add_swave(0.3) # BdG Hamiltonian with an s-wave gap

Ts = np.linspace(0.,0.3,13) # temperatures, in units of the hopping
Ds = [h.get_superfluid_weight(nk=20,T=T)[0,0] for T in Ts] # D_s^xx at each T
Tbkt = h.get_bkt_temperature(nk=20) # Nelson-Kosterlitz criterion

def draw(ax, c):
    ax.plot(Ts, Ds, "o-", color=c.blue, label=r"$D_s^{xx}$")
    ax.plot(Ts, 8*Ts/np.pi, "--", color=c.ink2, label=r"$8T/\pi$")
    ax.vlines(Tbkt, 0, 8*Tbkt/np.pi, color=c.orange, lw=1.2)
    ax.text(Tbkt+0.005, 0.05, r"$T_{\rm BKT}=%.2f$" % Tbkt, color=c.orange)
    ax.set_xlabel("temperature $T$ [$t$]")
    ax.set_ylabel(r"superfluid weight $D_s$")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper center", ncol=2)

style.save(draw, "superconductivity_superfluid")
