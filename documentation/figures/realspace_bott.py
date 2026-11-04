import style
from pyqula import geometry
import numpy as np

def haldane(): # Haldane model, C=1
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h.add_haldane(0.15) # Haldane coupling, clean gap 1.35
    return h

def kane_mele(): # Kane-Mele model with Rashba coupling, Z2 odd
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=True)
    h.add_kane_mele(0.1) # spin-orbit coupling
    h.add_rashba(0.05) # Rashba coupling, s_z is not conserved
    return h

Ws = np.arange(0.,10.01,0.5) # disorder strengths
nsamples = 20 # disorder configurations per strength

def scan(h0,method):
    """Index of every disordered sample, for every disorder strength"""
    b = np.zeros((len(Ws),nsamples))
    for (i,W) in enumerate(Ws):
        for s in range(nsamples):
            rng = np.random.default_rng(s)
            h = h0.copy()
            h.add_onsite(W*(rng.random(len(h.geometry.r))-0.5)) # Anderson disorder
            b[i,s] = getattr(h,method)()
    return b

B = scan(haldane().get_supercell(12),"get_bott_index") # 288 sites
Bs = scan(kane_mele().get_supercell(12),"get_spin_bott_index")

def draw(ax, c):
    for (a,b,label,col) in [(ax[0],B,"Bott index $B$",c.blue),
                            (ax[1],Bs,"spin Bott index $B_s$",c.orange)]:
        jitter = 0.1*(np.random.default_rng(0).random(b.shape)-0.5)
        a.scatter((Ws[:,None]+jitter).ravel(), b.ravel(), s=6, color=c.ink2, alpha=0.4)
        a.plot(Ws, b.mean(axis=1), color=col, lw=2, marker="o", ms=4)
        a.set_xlabel("disorder strength $W$ [$t$]")
        a.set_ylabel(label)
        a.set_ylim(-0.2,1.2)

style.save(draw, "realspace_bott", npanels=2)
