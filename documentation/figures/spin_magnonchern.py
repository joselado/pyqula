import numpy as np
import style
from pyqula import geometry
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(tij=[1.0,0.3]) # first and second neighbor hopping
h.add_kane_mele(0.15) # spin-orbit coupling, conserves Sz
h.add_zeeman([0.,0.,3.0]) # a field, so that the magnon has a gap
nk = 8 # the mean field and the magnons share this mesh
hmf = h.get_mean_field_hamiltonian(U=8.0,filling=0.5,mf="ferro",nk=nk)
nq = 12 # the mesh of Q
rs = [hmf.get_magnon_chern(bands=[b],nq=nq,nk=nk,full_output=True)
      for b in (0,1)] # the record of each magnon band
for r in rs: print("C =",r.chern,"gap",r.gap,"bound",r.bound,"max flux",r.max_flux)
# the Berry curvature, flux of a plaquette over its area, mean over the zone C
curv = [r.fluxes*nq**2/(2*np.pi) for r in rs]
vmax = max(np.max(np.abs(b)) for b in curv)
print("largest |curvature|",vmax)

def draw(axs, c):
    for (ax,b,r,name) in zip(axs,curv,rs,["lower","upper"]):
        im = ax.imshow(b.T, origin="lower", extent=[0,1,0,1], cmap=c.diverging,
                       vmin=-vmax, vmax=vmax)
        ax.set_xlabel(r"momentum $Q_1$ [$\mathbf{b}_1$]")
        ax.set_ylabel(r"momentum $Q_2$ [$\mathbf{b}_2$]")
        ax.set_title(f"{name} magnon band, C = {r.chern:.0f}")
    style.colorbar(im, axs[1], "Berry curvature")

style.save(draw, "spin_magnonchern", npanels=2)
