import numpy as np
import style
from pyqula import geometry
g = geometry.honeycomb_lattice() # create a honeycomb lattice
h = g.get_hamiltonian(has_spin=False) # get the Hamiltonian
h.add_sublattice_imbalance(0.3) # gapped honeycomb, gap 0.6

ws = np.linspace(0.,3.5,701) # frequencies
kw = dict(energies=ws,nk=256,T=1e-3,delta=1e-3) # broadening far below the mesh spacing
(ws,sg) = h.get_optical_conductivity(integration="grid",**kw) # sum over the k-mesh
(ws,st) = h.get_optical_conductivity(integration="triangle",**kw) # exact integral over triangles

def draw(ax, c):
    ax.plot(ws, 2*np.pi*sg[:,0,0].real, color=c.ink2, lw=0.8,
            label="grid, nk=256")
    ax.plot(ws, 2*np.pi*st[:,0,0].real, color=c.blue, lw=1.6,
            label="triangle, nk=256")
    ax.set_xlabel(r"frequency $\hbar\omega$ [$t$]")
    ax.set_ylabel(r"Re $\sigma_{xx}$ [$e^2/h$]")
    ax.legend(loc="upper right")

style.save(draw, "optical_triangle")
