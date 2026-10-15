import numpy as np
import style
from pyqula import specialhamiltonian
from pyqula import topology
h = specialhamiltonian.square_2OTI(delta=0.3) # quadrupole insulator, four sites per cell with pi flux
q = h.get_quadrupole_moment() # quadrupole moment from the nested Wilson loops
p = h.get_wannier_sector_polarization(loop=0,sector="+") # polarization along a2 of one Wannier sector
w = topology.z2_wannier_centers(h,full=True) # Wannier bands nu_1(k_2), from the Wilson loop along b1
gap = topology.wannier_gap(h) # distance of the Wannier bands to 0 and 1/2

def draw(ax, c):
    for nu in np.array(w[1:])/(2*np.pi): # one Wannier band per occupied band
        ax.scatter(w[0], nu, s=10, color=np.where(nu>0, c.blue, c.orange)) # the two sectors
    for y in [-0.5,0,0.5]: ax.axhline(y, color=c.ink2, lw=0.8, ls="--")
    ax.set_ylim(-0.55, 0.55)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"momentum $k_2$ [$|\mathbf{b}_2|$]")
    ax.set_ylabel(r"Wannier band $\nu_1$ [$a$]")
    ax.set_title(f"$q_{{xy}}$ = {q:.1f}, Wannier gap = {gap:.2f}")

style.save(draw, "topology_quadrupole")
