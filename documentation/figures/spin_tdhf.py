import numpy as np
import style
from pyqula import geometry
from pyqula.meanfield import VJinteraction
nk = 6                                  # the SCF and the magnon share this mesh
g = geometry.honeycomb_lattice()
scf = VJinteraction(g.get_hamiltonian(),U=3.0,V1=0.5,filling=0.5,
                     mf="antiferro",nk=nk,maxerror=1e-10)
hmf = scf.hamiltonian
print(hmf.get_goldstone_residual(nk=nk))          # 2e-10: the mode is exact
qs,es = hmf.get_magnon_bands(method="tdhf",nk=nk,nq=20,n=2)
x = (qs+1)/20 # position along the path, from Gamma through K, M, K to Gamma
E = np.sort(es.real.reshape(20,2), axis=1) # two branches at each q

def draw(ax, c):
    ax.plot(x[::2], E[:,0], "o-", color=c.blue, label="lowest mode")
    ax.plot(x[::2], E[:,1], "o-", color=c.orange, label="second mode")
    ax.set_xticks([1/3, 1/2, 2/3, 1], ["K", "M", "K'", r"$\Gamma$"])
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, None)
    ax.set_xlabel("momentum $Q$ along the path")
    ax.set_ylabel(r"magnon energy [$t$]")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)

style.save(draw, "spin_tdhf")
