import numpy as np
import style
from pyqula import geometry
from pyqula.bsetk.interaction import density_interaction
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian()
h.add_sublattice_imbalance(1.0) # a gapped semiconductor
W = density_interaction(h,Vr=lambda r1,r2: 0.8/np.sqrt((r1-r2).dot(r1-r2)+0.25))
nks = [6,9,12,15,18,21,24,30] # meshes up to 30x30 instead of nk=1024
eb = [] # binding energy of the lowest exciton on each mesh
for nk in nks:
    bse = h.get_bse(V=W,nk=nk,tda=True,solver="iterative",neig=4)
    eb.append(bse.get_binding_energies()[0].real)
    print(nk, bse.get_energies()[0].real, eb[-1]) # lowest exciton, and how far below the gap

nk = nks[-1] # weight |A(k)|^2 of the lowest exciton on the finest mesh
weight = np.zeros(len(bse.pairs.kpoints))
for A in bse.amplitudes:
    np.add.at(weight, bse.pairs.kindex, np.abs(A)**2)
weight = weight/len(bse.amplitudes) # average over the four states
k1 = np.arange(nk)/nk # the mesh, in units of the reciprocal lattice vectors

def draw(ax, c):
    ax[0].plot(nks, eb, "o-", color=c.blue)
    ax[0].set_xlabel("k-mesh $n_k$ (an $n_k\\times n_k$ mesh)")
    ax[0].set_ylabel(r"binding energy [$t$]")
    im = ax[1].pcolormesh(k1, k1, weight.reshape(nk,nk).T, cmap=c.cmap,
                          shading="nearest")
    style.colorbar(im, ax[1], r"weight $\sum_{vc}|A_{vc}(k)|^2$")
    ax[1].text(1/3+0.05, 1/3-0.05, "K", color=c.blue, va="top")
    ax[1].text(2/3+0.05, 2/3-0.05, "K'", color=c.blue, va="top")
    ax[1].set_aspect("equal")
    ax[1].set_xlabel("$k_1$")
    ax[1].set_ylabel("$k_2$")

style.save(draw, "excitons_largek", npanels=2)
