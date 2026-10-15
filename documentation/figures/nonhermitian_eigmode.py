import numpy as np
import style
from pyqula import geometry
n = 20
g = geometry.chain().get_supercell(n)
h = g.get_hamiltonian(has_spin=False,non_hermitian=True,tij=[-1.])
h.add_onsite(lambda r: 0.6j*np.cos(2.*np.pi*r[0]/n))
(ks,es) = h.get_bands(kpath=[[0.,0.,0.]],write=False)

imax = np.argmax(es.imag) # the most amplified state
(x,y,d) = h.get_ldos(e=es[imax].imag,delta=1e-2,eigmode="imag",nrep=1)
print("it lives at x =",x[np.argmax(d)])

rates = np.unique(es.imag.round(4)) # the different gain rates
ldos = [h.get_ldos(e=w,delta=1e-2,eigmode="imag",nrep=1)[2] for w in rates]
ldos = [d/d.max() for d in ldos] # each row normalized to its largest value

def draw(ax, c):
    for (w,d) in zip(rates,ldos):
        sc = ax.scatter(x, w+0*x, c=d, s=70, marker="s", cmap=c.cmap, vmin=0, vmax=1)
    style.colorbar(sc, ax, "LDOS at this gain rate")
    ax.plot(x, 0.6*np.cos(2*np.pi*x/n), "--", color=c.blue, lw=1.5, label="gain profile")
    ax.set_xlabel("position $x$")
    ax.set_ylabel(r"gain rate Im$\,E$ [$t$]")
    ax.set_ylim(-0.65, 0.65)
    ax.legend(loc="upper right")

style.save(draw, "nonhermitian_eigmode")
