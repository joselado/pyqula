import numpy as np
import style
from matplotlib.colors import LogNorm
from pyqula import geometry
g = geometry.chain()
h = g.get_hamiltonian(has_spin=True) # 1D chain at half filling
qs,ws,gammas = h.get_plasmon_bands(V1=0.6,qpath=[[0.3,0.,0.],[0.4,0.,0.],[0.5,0.,0.]],nq=3,
                                    energies=np.linspace(0.,1.,100),delta=2e-2,nk=2000)

# the RPA charge response over (q,omega), as a map
qmap = np.linspace(0.,0.5,21) # momentum transfer
es = np.linspace(0.,4.,120) # frequencies
chimap = [h.get_densitychi_RPA(V1=0.6,q=[q,0.,0.],energies=es,delta=2e-2,nk=1000)[1][:,0,0].imag
          for q in qmap]

def draw(ax, c):
    im = ax.pcolormesh(qmap, es, np.clip(np.array(chimap).T, 1e-2, None),
                       cmap=c.cmap, norm=LogNorm(vmin=1e-2),
                       shading="nearest", rasterized=True)
    style.colorbar(im, ax, r"Im $\chi_{\rm RPA}(q,\omega)$")
    qline = np.linspace(0., 0.5, 100)
    ax.plot(qline, 4*np.sin(np.pi*qline), "--", color=c.green, lw=1.2)
    ax.plot(qline, 2*np.sin(2*np.pi*qline), "--", color=c.green, lw=1.2,
            label="continuum edges")
    ax.scatter(np.array([0.3,0.4,0.5])[qs.astype(int)], ws, s=50, facecolors="none",
               edgecolors=c.blue, lw=1.5, label="plasmon-band poles")
    ax.set_xlabel(r"momentum transfer $q$ [$2\pi/a$]")
    ax.set_ylabel(r"frequency $\omega$ [$t$]")
    ax.set_ylim(0, 4)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)

style.save(draw, "spin_plasmon")
