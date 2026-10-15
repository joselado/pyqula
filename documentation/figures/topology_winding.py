import numpy as np
import style
from pyqula import geometry
g = geometry.bichain() # two-site chain with a sublattice
def fun(r1,r2): # intracell hopping 0.5, intercell hopping 1
    dr = r1-r2
    if abs(dr.dot(dr)-1.)>1e-4: return 0. # first neighbors only
    return 0.5 if abs((r1[0]+r2[0])/2.)<1e-4 else 1.0 # the intracell bond is centered at x=0
h = g.get_hamiltonian(fun=fun,has_spin=False) # SSH chain
W = h.get_winding_number() # winding number, with the sublattice operator
hk = h.get_hk_gen() # Bloch Hamiltonian H(k)
ks = np.linspace(0.,1.,200) # momenta across the Brillouin zone
q = np.array([hk([k,0.,0.])[0,1] for k in ks]) # off-diagonal block q(k), a number here

def draw(ax, c):
    ax.plot(q.real, q.imag, color=c.blue)
    for i in [25,75,125,175]: # arrows in the direction of growing k
        ax.annotate("", xy=(q[i+1].real,q[i+1].imag), xytext=(q[i].real,q[i].imag),
                    arrowprops=dict(arrowstyle="-|>", color=c.blue, mutation_scale=25))
    ax.annotate("$k=0$", (q[0].real,q[0].imag), xytext=(8,0), textcoords="offset points")
    ax.scatter([0], [0], marker="x", s=80, color=c.ink)
    ax.set_aspect("equal")
    ax.set_xlabel(r"Re $q(k)$ [$t$]")
    ax.set_ylabel(r"Im $q(k)$ [$t$]")
    ax.set_title(f"SSH chain, W = {W}")

style.save(draw, "topology_winding")
