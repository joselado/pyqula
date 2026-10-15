import numpy as np
import style
from pyqula import geometry
from pyqula import rkky
g = geometry.chain()
h = g.get_hamiltonian()
h.add_onsite(1.)
m = rkky.rkky_map(h,n=10,mode="LR",nk=200) # linear-response RKKY vs distance
x,e = m[:,0],m[:,3]
x,e = x[x>0],e[x>0] # keep one side, without the reference site

def draw(ax, c):
    col = [c.orange if v>0 else c.blue for v in e]
    ax.vlines(x, 0, e, color=col, lw=3)
    ax.scatter(x, e, color=col, s=40, zorder=3)
    ax.axhline(0, color=c.ink, lw=0.6, alpha=0.4)
    ax.set_xlabel("distance [$a$]")
    ax.set_ylabel(r"RKKY coupling $J_{\rm RKKY}$")
    ax.text(0.98, 0.95, "ferromagnetic", color=c.orange, ha="right",
            va="top", transform=ax.transAxes)
    ax.text(0.98, 0.05, "antiferromagnetic", color=c.blue, ha="right",
            va="bottom", transform=ax.transAxes)

style.save(draw, "charge_rkky")
