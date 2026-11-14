import numpy as np
import style
from pyqula import geometry

g = geometry.honeycomb_lattice() # unit cell of graphene
h = g.get_hamiltonian(has_spin=False) # spinless, first-neighbor hopping t=1
n = 30 # period of the modulation, in unit cells along a1 and a2
hs = h.get_supercell([n,n]) # the ultracell, written as a supercell
B = np.linalg.inv(np.array([g.a1,g.a2,g.a3]).T) # B@r are the coordinates along a1, a2
V0 = 0.1 # amplitude of each plane wave, in units of t
def V(r): # a potential with the period of the ultracell and threefold symmetry
    x1,x2 = 2*np.pi*(B@r)[0:2]/n
    return V0*(np.cos(x1) + np.cos(x2) + np.cos(x1-x2))
hs.add_onsite(V) # add it to the ultracell
u = h.get_ultracell(hs) # hs solved in the Bloch states of graphene
(x,y,d) = u.get_ldos(e=0.0,delta=0.03,nk=2) # LDOS at the Dirac energy
dn = u.get_vev(nk=2) - 0.5 # extra charge on each site, Fermi level at zero
(Q,f) = u.get_harmonics(dn) # its Fourier components, per unit-cell site

v = np.array([V(r) for r in hs.geometry.r]) # the potential on each site
print("potential from",v.min(),"to",v.max())
print("correlation of the LDOS with |V|:",np.corrcoef(d,np.abs(v))[0,1])
print("LDOS where V>0.25:",d[v>0.25].mean(),", V<-0.13:",d[v<-0.13].mean(),
      ", |V|<0.02:",d[np.abs(v)<0.02].mean())
print("extra charge from",dn.min(),"to",dn.max(),", total",dn.sum())
print("correlation of the extra charge with -V:",np.corrcoef(dn,-v)[0,1])
order = np.argsort(-np.abs(f[:,0]))[0:8]
print("largest harmonics of the extra charge on sublattice A, Q*n and |f|:")
for i in order: print("  ",np.round(Q[i,0:2]*n).astype(int),abs(f[i,0]))

A1,A2 = np.array(hs.geometry.a1),np.array(hs.geometry.a2) # vectors of the ultracell
tile = lambda z: np.concatenate([z]*4) # four copies of the ultracell, 2x2
X = np.concatenate([x+i*A1[0]+j*A2[0] for i in (0,1) for j in (0,1)])
Y = np.concatenate([y+i*A1[1]+j*A2[1] for i in (0,1) for j in (0,1)])

def draw(axs, c):
    m = np.max(np.abs(v))
    s = axs[0].scatter(X, Y, c=tile(v), s=2, lw=0, cmap=c.diverging, vmin=-m, vmax=m)
    style.colorbar(s, axs[0], r"potential $V(\mathbf{r})$ [$t$]")
    s = axs[1].scatter(X, Y, c=tile(d), s=2, lw=0, cmap=c.cmap, vmax=np.percentile(d,99.5))
    style.colorbar(s, axs[1], r"LDOS at $\omega=0$")
    for ax in axs:
        ax.set_aspect("equal")
        ax.set_axis_off()

style.save(draw, "ultracell_ldos", npanels=2)
