# h2 and W as in the section above
import numpy as np
from matplotlib.colors import PowerNorm
import style
from pyqula import geometry
from pyqula.bsetk.interaction import density_interaction
g = geometry.honeycomb_lattice()
h2 = g.get_hamiltonian()
h2.add_sublattice_imbalance(1.0) # a gapped semiconductor
W = density_interaction(h2,Vr=lambda r1,r2: 0.8/np.sqrt((r1-r2).dot(r1-r2)+0.25))
ws = np.linspace(1.7,3.0,300) # energy grid
opts = dict(V=W,nq=40,nk=18,nv=2,nc=2,energies=ws,delta=0.05) # nq=40 for a smooth map
qs,es,ds = h2.get_exciton_kdos_bands(**opts) # exciton density of states
qs,es,ss = h2.get_exciton_structure_factor(**opts) # what EELS measures
nq = qs.max()+1
x = (np.arange(nq)+1)/nq # position along the path, from Gamma through K, M, K' to Gamma
D = ds.reshape(nq,-1).T ; S = ss.reshape(nq,-1).T # (energy, q-point)

def draw(ax, c):
    for a,M,t in zip(ax,[D,S],["exciton DOS",r"structure factor $S(Q,\omega)$"]):
        # a square-root scale, so that the bound excitons are visible
        # next to the continuum, which holds most of the states
        a.pcolormesh(x,ws,M,cmap=c.cmap,shading="nearest",
                norm=PowerNorm(0.5,vmin=0.,vmax=M.max()))
        a.set_xticks([1/3,1/2,2/3,1],["K","M","K'",r"$\Gamma$"])
        a.set_xlabel("center-of-mass momentum $Q$")
        a.set_title(t)
    ax[0].set_ylabel(r"energy [$t$]")

style.save(draw, "excitons_spectral", npanels=2)
