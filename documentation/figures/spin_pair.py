import numpy as np
import style
from matplotlib.colors import LogNorm
from pyqula import geometry
from pyqula.meanfield import VJinteraction
nk = 24                                 # the SCF and the magnon share this mesh
g = geometry.honeycomb_lattice()
hmf = VJinteraction(g.get_hamiltonian(),U=3.0,V1=0.5,filling=0.5,
                     mf="antiferro",nk=nk,mix=0.5,maxerror=1e-10).hamiltonian
qs,ws,gammas = hmf.get_magnon_bands(method="pair",nq=90,
                    energies=np.linspace(1e-3,3.,400),delta=1e-3,nk=nk)
es,chi = hmf.get_transverse_spinchi(energies=np.linspace(0.,2.,300),
                    q=[0.1,0.,0.],delta=1e-2,nk=nk)   # (Sx,Sy,Sz) x site chi(w)

# the same response along the path of the magnon bands, as a map
nq = 90
qpath = g.get_kpath(None,nk=nq) # the default path, from Gamma through K, M, K' to Gamma
chimap = [np.trace(hmf.get_transverse_spinchi(energies=es,q=q,delta=1e-2,nk=nk)[1],
                   axis1=1,axis2=2).imag for q in qpath]
x = (np.arange(nq)+1)/nq # position along the path
lowest = np.array([ws[qs==i].min() for i in range(nq)]) # lowest pole at each q

def draw(ax, c):
    im = ax.pcolormesh(x, es, np.clip(np.array(chimap).T, 1e-2, None),
                       cmap=c.cmap, norm=LogNorm(vmin=1e-2),
                       shading="nearest", rasterized=True)
    style.colorbar(im, ax, r"Im Tr $\chi_\perp(q,\omega)$")
    ax.scatter((qs+1)/nq, ws, s=1, color=c.ink2, alpha=0.5, lw=0,
               label="other poles")
    ax.scatter(x, lowest, s=14, facecolors="none", edgecolors=c.blue,
               lw=0.8, label="lowest pole")
    ax.set_xticks([1/3, 1/2, 2/3, 1], ["K", "M", "K'", r"$\Gamma$"])
    ax.set_xlabel("momentum $q$ along the path")
    ax.set_xlim(0.5/nq, 1+0.5/nq)
    ax.set_ylim(0, 2)
    ax.set_ylabel(r"frequency $\omega$ [$t$]")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              markerscale=2)

style.save(draw, "spin_pair")
