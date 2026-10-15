import numpy as np
import style
from pyqula import geometry
from pyqula import topology
g = geometry.honeycomb_lattice() # create honeycomb lattice
h = g.get_hamiltonian() # create hamiltonian of the system (spinful by default)
h.add_haldane(0.2) # Add Haldane coupling
h.shift_fermi(0.3) # put the Fermi level safely mid-gap (gap is [-0.9,0.9])

Q = h.get_quantum_geometric_tensor(k=[0.1,0.2,0.],occ_idxs=[0,1]) # at a k-point
g_metric = h.get_quantum_metric(k=[0.1,0.2,0.],occ_idxs=[0,1]) # quantum metric only

# non-Abelian tensor of the two occupied bands, in the orbital basis
Qna = topology.quantum_geometric_tensor(h,k=[0.1,0.2,0.],occ_idxs=[0,1],
        non_abelian=True)

# along a k-path, and integrated over the BZ: the integral of the Berry
# curvature part reproduces h.get_chern()
inds,gpath,omegapath = topology.quantum_geometric_tensor_path(h,occ_idxs=[0,1])
C = topology.chern_from_qgt(h,nk=20,occ_idxs=[0,1])

# the tensor on a k-mesh, in Cartesian momenta
ks,Qk = topology.quantum_geometric_tensor_mesh(h,nk=60,occ_idxs=[0,1],
        coordinates="cartesian")
trg = (Qk[:,0,0]+Qk[:,1,1]).real # trace of the quantum metric, g_xx+g_yy
omega = -2*Qk[:,0,1].imag # Berry curvature
ks = [ks+[i,j,0] for i in range(-2,2) for j in range(-2,2)] # 16 copies of the BZ
kxy = 2*np.pi*np.array([k[:,0:1]*g.b1+k[:,1:2]*g.b2 for k in ks]).reshape(-1,3) # Cartesian
trg,omega = np.tile(trg,16),np.tile(omega,16)

def draw(ax, c):
    m = np.max(np.abs(omega))
    p0 = ax[0].tripcolor(kxy[:,0], kxy[:,1], trg, cmap=c.cmap, shading="gouraud")
    p1 = ax[1].tripcolor(kxy[:,0], kxy[:,1], omega, cmap=c.diverging,
                         vmin=-m, vmax=m, shading="gouraud")
    style.colorbar(p0, ax[0], r"quantum metric $g_{xx}+g_{yy}$ [$d^2$]")
    style.colorbar(p1, ax[1], r"Berry curvature $\Omega_{xy}$ [$d^2$]")
    for axi in ax:
        axi.set_xlim(-3.5, 3.5)
        axi.set_ylim(-3.5, 3.5)
        axi.set_aspect("equal")
        axi.set_xlabel(r"momentum $k_x$ [$1/d$]")
    ax[0].set_ylabel(r"momentum $k_y$ [$1/d$]")

style.save(draw, "geometry_qgt", npanels=2)
