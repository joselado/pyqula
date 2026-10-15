import numpy as np
import style
from pyqula import geometry
from pyqula import classicalspin
from pyqula.classicalspintk.align import most_perp_basis
np.random.seed(1) # fixed seed, so that every run gives the same figure

g = geometry.triangular_lattice() # geometrically frustrated lattice
g = g.get_supercell(3)

sm = classicalspin.SpinModel(g) # classical spin model on this geometry
sm.add_heisenberg(Jij=[1.0]) # first-neighbor antiferromagnetic exchange
sm.minimize_energy(tries=10) # multistart local minimization

mx,my,mz = sm.get_magnetization() # per-site magnetization components
m = most_perp_basis(np.array([mx,my,mz]).T) # rotate the spins into the xy plane

def draw(ax, c):
    phi = np.arctan2(m[:,1], m[:,0]) # in-plane angle of each spin
    sub = np.round((phi - phi[0])/(2*np.pi/3)).astype(int) % 3 # its sublattice
    for i in range(-2,3): # draw copies of the supercell
        for j in range(-2,3):
            r = g.r + i*g.a1 + j*g.a2
            for s in range(3):
                k = (sub == s) & (abs(r[:,0]) < 3.6) & (abs(r[:,1]) < 3.0)
                ax.quiver(r[k,0], r[k,1], m[k,0], m[k,1], color=c.series[s],
                          pivot="middle", scale=1.7, scale_units="xy", width=0.009)
    ax.set_aspect("equal")
    ax.set_xlim(-3.7, 3.7) ; ax.set_ylim(-3.2, 3.2)
    ax.set_xlabel("$x$") ; ax.set_ylabel("$y$")

style.save(draw, "classicalspins_120")
