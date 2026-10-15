import numpy as np
import style
from matplotlib.colors import LogNorm
from pyqula import geometry
from pyqula.meanfield import VJinteraction
nk = 6                                  # the SCF and the magnon share this mesh
g = geometry.honeycomb_lattice()
hmf = VJinteraction(g.get_hamiltonian(),U=3.0,V1=0.5,filling=0.5,
                     mf="antiferro",nk=nk,maxerror=1e-10).hamiltonian
qs,ws,gammas = hmf.get_magnon_bands(method="pair",nq=20,
                    energies=np.linspace(1e-3,3.,400),delta=1e-3,nk=nk)
es,chi = hmf.get_transverse_spinchi(energies=np.linspace(0.,2.,100),
                    q=[0.1,0.,0.],delta=1e-2,nk=nk)   # (Sx,Sy,Sz) x site chi(w)

# the same response along the path of the magnon bands, as a map
qpath = g.get_kpath(None,nk=20) # the default path, from Gamma through K, M, K to Gamma
chimap = [np.trace(hmf.get_transverse_spinchi(energies=es,q=q,delta=1e-2,nk=nk)[1],
                   axis1=1,axis2=2).imag for q in qpath]
x = (np.arange(20)+1)/20 # position along the path

def draw(ax, c):
    im = ax.pcolormesh(x, es, np.clip(np.array(chimap).T, 1e-2, None),
                       cmap=c.cmap, norm=LogNorm(vmin=1e-2),
                       shading="nearest", rasterized=True)
    style.colorbar(im, ax, r"Im Tr $\chi_\perp(q,\omega)$")
    ax.scatter((qs+1)/20, ws, s=12, facecolors="none",
               edgecolors=c.blue, lw=0.8, label="poles of the magnon bands")
    ax.set_xticks([1/3, 1/2, 2/3, 1], ["K", "M", "K'", r"$\Gamma$"])
    ax.set_xlabel("momentum $q$ along the path")
    ax.set_xlim(0.025, 1.025)
    ax.set_ylim(0, 2)
    ax.set_ylabel(r"frequency $\omega$ [$t$]")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0))

style.save(draw, "spin_pair")
