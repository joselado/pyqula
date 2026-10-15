import numpy as np
import style
from pyqula import geometry
from pyqula import kdos
g = geometry.honeycomb_lattice() # create honeycomb lattice
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_haldane(0.05) # Add Haldane coupling
C = h.get_chern() # Chern number
nk = 100 # k-points per direction
(kx,ky,b) = h.get_berry_curvature(nk=nk) # Berry curvature, a window of two zones
print("mean/(2 pi) =",np.mean(b)/(2*np.pi)) # the Chern number
b = b.reshape(nk,nk).T # as an image, ky along the rows
(Lx,Ly) = (-2*kx.min(),-2*ky.min()) # the window is centered at Gamma
b = np.concatenate([b[:,nk//2:],b,b[:,:nk//2]],axis=1) # periodic, shown twice as wide
(dx,dy) = (Lx/nk/2,Ly/nk/2) # half a pixel
phi = np.pi/2 + np.pi/3*np.arange(7) # corners of the Brillouin zone, the K points

def draw(ax, c):
    im = ax.imshow(b, origin="lower", extent=[-Lx-dx,Lx-dx,-Ly/2-dy,Ly/2-dy], cmap=c.diverging,
                   vmin=-200, vmax=200, interpolation="bicubic")
    ax.plot(2/3*np.cos(phi), 2/3*np.sin(phi), color="gray", lw=1)
    style.colorbar(im, ax, r"Berry curvature $\Omega(\mathbf{k})$")
    ax.set_xlabel(r"momentum $k_x$ [$2\pi/a$]")
    ax.set_ylabel(r"momentum $k_y$ [$2\pi/a$]")
    ax.set_title(f"Haldane model, C = {C:.0f}")

style.save(draw, "topology_chern")
