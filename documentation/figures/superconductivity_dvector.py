import style
from pyqula import geometry
g = geometry.chain() # geometry of the 1D model
h = g.get_hamiltonian() # generate the Hamiltonian
h.setup_nambu_spinor() # initialize the Nambu basis
h0 = h.copy() # keep the normal chain for the unitary state
h.add_pairing(delta=0.3,mode="pwave",d=[1.,1j,0.]) # up-up pairing
q = h.get_dvector_non_unitarity(nk=20) # -> [[0., 0., 0.36]]
h0.add_pairing(delta=0.3,mode="pwave",d=[1.,0.,0.]) # real d-vector, unitary
q0 = h0.get_dvector_non_unitarity(nk=20) # -> zero
bands = [h0.get_bands(operator="sz"), h.get_bands(operator="sz")] # (k,e,sz)
titles = [r"$\vec d=(1,0,0)$, $q_z=%.2f$" % q0[0][2],
          r"$\vec d=(1,i,0)$, $q_z=%.2f$" % q[0][2]]

def draw(axs, c):
    for (ax, (k,e,sz), t) in zip(axs, bands, titles):
        sc = ax.scatter(k, e, c=sz, s=4, cmap=c.diverging, vmin=-1, vmax=1)
        style.kpath(ax, k, ["0", r"$\pi$", r"$2\pi$"])
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel("momentum $k$")
        ax.set_title(t)
    axs[0].set_ylabel("energy [$t$]")
    style.colorbar(sc, axs[1], r"spin $\langle S_z\rangle$")

style.save(draw, "superconductivity_dvector", npanels=2)
