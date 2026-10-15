import numpy as np
import style
from pyqula import specialhamiltonian

h = specialhamiltonian.iwave_altermagnet(J=0.3)   # or dwave_/fwave_/gwave_

# largest |sigma_spin| over every component of each order l
orders = h.get_nonlinear_drude_orders(lmax=6, nk=48, T=0.02, mu=-5.5)
# -> zero to machine precision for l = 0..4, nonzero at l = 5

# one component, and every component of a given order
s = h.get_nonlinear_drude_conductivity(field="yyyyy", current="x", mu=-5.5)
c = h.get_nonlinear_drude_components(5, nk=48, mu=-5.5)   # {"x^l1 y^l2;b": value}

# the same for every model, with mu 0.5 above the bottom of its band
waves = {"p":-3.5, "d":-3.5, "f":-5.5, "g":-3.5, "i":-5.5} # wave: mu
table = [specialhamiltonian.xwave_magnet(wave=w,J=0.3,t=-1.).get_nonlinear_drude_orders(
         lmax=6, nk=48, T=0.02, mu=mu) for (w,mu) in waves.items()]
logs = np.log10(np.maximum(np.array(table), 1e-16)) # zeros drawn at 1e-16

def draw(ax, cl):
    m = ax.imshow(logs, cmap=cl.cmap, vmin=-16, vmax=0, aspect="auto")
    style.colorbar(m, ax, r"$\log_{10}$ max $|\sigma_{\rm spin}|$")
    ax.set_xticks(range(7))
    ax.set_yticks(range(len(waves)), [w+"-wave" for w in waves])
    ax.set_xlabel("order $l$ of the electric field")
    ax.spines[:].set_visible(False)

style.save(draw, "altermagnets_orders")
