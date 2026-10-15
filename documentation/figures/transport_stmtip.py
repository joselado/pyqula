import numpy as np
import style
from pyqula import geometry
from pyqula.transporttk.localprobe import LocalProbe

g = geometry.chain()
h = g.get_hamiltonian() ; h.shift_fermi(1.) ; h.add_swave(0.1) # SC sample
lead = geometry.chain().get_hamiltonian() ; lead.shift_fermi(1.) ; lead.add_swave(0.1) # SC probe
lp = LocalProbe(h,lead=lead,delta=1e-3)
lp.T = 0.3 # reference transparency
es = np.linspace(0.1,0.4,16) # 16 voltages from Delta up, instead of the single 0.25
G = lp.didv(energies=es,nmax=4,nmax_max=12,tol=5e-2) # routed through Keldysh automatically
k = lp.get_kappa(energies=es,nmax=4,nmax_max=12,tol=5e-2)
for (e,Ge,ke) in zip(es,G,k): print(e,Ge,ke)

def draw(ax, c):
    for a in ax:
        for (x,name) in [(0.1,r"$\Delta$"),(0.2,r"$2\Delta$")]:
            a.axvline(x, color=c.ink2, lw=0.8, ls="--")
            a.text(x, 1.0, " "+name, color=c.ink2, va="top",
                   transform=a.get_xaxis_transform())
        a.set_xlabel(r"bias voltage $eV$ [$t$]")
    ax[0].plot(es, G, "o-", color=c.blue)
    ax[0].set_ylabel(r"conductance $dI/dV$ [$e^2/h$]")
    ax[1].plot(es, k, "o-", color=c.orange)
    for kappa in [1,2]: # single quasiparticle, and Andreev reflection
        ax[1].axhline(kappa, color=c.ink2, lw=0.8, ls=":")
    ax[1].set_ylabel(r"decay rate $\kappa$")

style.save(draw, "transport_stmtip", npanels=2)
