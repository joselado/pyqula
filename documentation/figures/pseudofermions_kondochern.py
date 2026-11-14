import numpy as np
import style
from pyqula import geometry
from pyqula.kondolattice import KondoLatticeHamiltonian
hc = geometry.honeycomb_lattice().get_hamiltonian(has_spin=True)
hc.add_haldane(0.2) # breaks time reversal of the conduction electrons
h = KondoLatticeHamiltonian(hc) # two moments per unit cell
h2 = h.get_mean_field_hamiltonian(J=4.0, filling=0.5, nk=12,
        mf=([1.0+0j]*2, [0.0]*2))
nq = 12 # the mesh of q
r = h2.get_moment_magnon_chern((0.3, 1.43), U=-4.0, nk=8, nq=nq,
        full_output=True)
print("C =",r.chern,"gap",r.gap,"bound",r.bound,"max flux",r.max_flux)
print("energies",r.energies.min(),r.energies.max())
e = r.energies[:,:,0] # energy of the magnon of the moments on the mesh
b = r.fluxes*nq**2/(2*np.pi) # Berry curvature, mean over the zone C
vmax = np.max(np.abs(b))
print("largest |curvature|",vmax)

def draw(axs, c):
    im = axs[0].imshow(e.T, origin="lower", extent=[0,1,0,1], cmap=c.cmap)
    style.colorbar(im, axs[0], r"magnon energy [$t$]")
    im = axs[1].imshow(b.T, origin="lower", extent=[0,1,0,1], cmap=c.diverging,
                       vmin=-vmax, vmax=vmax)
    style.colorbar(im, axs[1], "Berry curvature")
    axs[1].set_title(f"C = {r.chern:.0f}")
    for ax in axs:
        ax.set_xlabel(r"momentum $q_1$ [$\mathbf{b}_1$]")
        ax.set_ylabel(r"momentum $q_2$ [$\mathbf{b}_2$]")

style.save(draw, "pseudofermions_kondochern", npanels=2)
