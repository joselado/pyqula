import numpy as np
import style
from pyqula import geometry
g = geometry.chain().get_supercell(4,store_primal=True) # supercell of a chain
h = g.get_hamiltonian(has_spin=False,non_hermitian=True) # non-Hermitian Hamiltonian
h.add_onsite(lambda r: -0.3j*(1.+np.cos(np.pi*r[0]/2.))) # modulated loss
kpath = g.get_unfolded_kpath(nk=400) # primitive k-path, in supercell coordinates
(k,e,d) = h.get_kdos_bands(operator="unfold",kpath=kpath,
        biorthogonal=True) # unfolded Green's-function spectral function
es = np.linspace(-3.,3.,1201) # a finer energy grid than the default, for the figure
(k,e,d) = h.get_kdos_bands(operator="unfold",kpath=kpath,energies=es,biorthogonal=True)
(k0,e0,d0) = h.get_kdos_bands(operator="unfold",kpath=kpath,energies=es) # the default one

nk, ne = len(kpath), len(es) # the arrays are flat, k-point by k-point
K = 2*np.pi*k.reshape(nk,ne)/k.max() # momentum of the primitive chain
E = e.reshape(nk,ne)

def draw(axs, c):
    for (ax,dd,title) in zip(axs, [d0,d], ["right eigenvectors (default)",
                                         "Green's function, biorthogonal=True"]):
        m = ax.pcolormesh(K, E, dd.reshape(nk,ne), cmap=c.cmap, vmin=0,
                          vmax=dd.max(), shading="nearest")
        style.colorbar(m, ax, r"spectral function $A(k,\omega)$")
        style.kpath(ax, K.ravel(), ["0", r"$\pi$", r"$2\pi$"])
        ax.set_xlabel("momentum $k$")
        ax.set_ylabel(r"energy $\omega$ [$t$]")
        ax.set_title(title)

style.save(draw, "nonhermitian_kdos", npanels=2)
