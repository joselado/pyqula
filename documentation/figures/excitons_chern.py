import numpy as np
import style
from pyqula import geometry
from pyqula.multihopping import MultiHopping
from pyqula.bsetk.interaction import density_interaction
phi = 0.3 # flux through each up triangle
H = np.zeros((3,3),dtype=np.complex128) # one up triangle, a molecule
for (i,j) in [(0,1),(1,2),(2,0)]:
    H[j,i] = -np.exp(1j*phi/3.) # hopping with a Peierls phase
    H[i,j] = np.conj(H[j,i])
H += 0.5*np.identity(3) # one level below the Fermi energy, two above
g = geometry.kagome_lattice()
h = g.get_hamiltonian(has_spin=False)
h.set_multihopping(MultiHopping({(0,0,0):H})) # only the hopping in the cell
W = density_interaction(h,V1=1.0) # density interaction between neighbors
nq = 12 # the mesh of Q
r = h.get_exciton_chern(bands=[0],nq=nq,V=W,nk=6,tda=True,full_output=True)
print("C =",r.chern,"gap",r.gap,"bound",r.bound,"max flux",r.max_flux)
print("energies",r.energies.min(),r.energies.max())
e = r.energies[:,:,0] # energy of the lowest exciton on the mesh
b = r.fluxes*nq**2/(2*np.pi) # Berry curvature, mean over the zone C
vmax = np.max(np.abs(b))
print("largest |curvature|",vmax)

def draw(axs, c):
    im = axs[0].imshow(e.T, origin="lower", extent=[0,1,0,1], cmap=c.cmap)
    style.colorbar(im, axs[0], r"exciton energy [$t$]")
    im = axs[1].imshow(b.T, origin="lower", extent=[0,1,0,1], cmap=c.diverging,
                       vmin=-vmax, vmax=vmax)
    style.colorbar(im, axs[1], "Berry curvature")
    axs[1].set_title(f"C = {r.chern:.0f}")
    for ax in axs:
        ax.set_xlabel(r"momentum $Q_1$ [$\mathbf{b}_1$]")
        ax.set_ylabel(r"momentum $Q_2$ [$\mathbf{b}_2$]")

style.save(draw, "excitons_chern", npanels=2)
