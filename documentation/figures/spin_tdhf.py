import numpy as np
import style
from pyqula import geometry
from pyqula.meanfield import VJinteraction
nk = 24                                 # the SCF and the magnon share this mesh
g = geometry.honeycomb_lattice()
scf = VJinteraction(g.get_hamiltonian(),U=3.0,V1=0.5,filling=0.5,
                     mf="antiferro",nk=nk,mix=0.5,maxerror=1e-10)
hmf = scf.hamiltonian
print(hmf.get_goldstone_residual(nk=nk))          # 3e-10: zero to the SCF tolerance
qs,es = hmf.get_magnon_bands(method="tdhf",nk=nk,nq=90,n=2)
x = (np.arange(90)+1)/90 # position along the path, from Gamma through K, M, K' to Gamma
E = np.sort(es.real.reshape(90,2), axis=1) # two branches at each q
print(E[[0,29,44,89]]) # first point, K, M and Gamma

def draw(ax, c):
    ax.plot(x, E[:,0], "o-", ms=3, color=c.blue, label="lowest mode")
    ax.plot(x, E[:,1], "o-", ms=3, color=c.orange, label="second mode")
    ax.set_xticks([1/3, 1/2, 2/3, 1], ["K", "M", "K'", r"$\Gamma$"])
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, None)
    ax.set_xlabel("momentum $Q$ along the path")
    ax.set_ylabel(r"magnon energy [$t$]")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)

style.save(draw, "spin_tdhf")
