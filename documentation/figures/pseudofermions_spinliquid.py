import numpy as np
import style
from pyqula import geometry
from pyqula.spinon import SpinonHamiltonian

g0 = geometry.triangular_lattice()
g = g0.get_supercell((2,1)) # two sites, the cell of the pi-flux pattern
a = g.get_hamiltonian(has_spin=True) # first-neighbor hoppings, all +1
ansatz = dict()
for (d,m) in a.get_dict().items(): # put the signs of the pi-flux pattern
    m = np.array(m)
    for i in range(2):
        for j in range(2):
            dr = g.r[j] + d[0]*g.a1 + d[1]*g.a2 - g.r[i] # the bond
            if abs(dr[1])<1e-6: continue # a bond along a1 keeps +1
            p = i if dr[1]>0 else i + int(round(dr[0]-0.5)) # its lower end
            m[2*i:2*i+2,2*j:2*j+2] *= (-1)**p
    ansatz[d] = m

h = SpinonHamiltonian(g)
h2 = h.get_mean_field_hamiltonian(J1=1.0,J2=0.125,decoupling="bond",
        ansatz=ansatz,gutzwiller=2.0,nk=20,T=0.02)

energies = np.linspace(0.,3.,100)
path = [[0.,0.],[1/3,1/3],[1/2,0.],[0.,0.]] # Gamma, K, M, Gamma of the lattice
qs = np.concatenate([np.linspace(path[i],path[i+1],20,endpoint=False)
                     for i in range(3)])
x = g.r[:,0] - g.r[0,0] # position of the two sites along a1
def structure_factor(RPA):
    out = []
    for (q1,q2) in qs: # momentum in units of the vectors of the lattice
        (es,chi) = h2.get_spinchi_full(q=[2*q1,q2,0.],energies=energies,
                delta=0.05,nk=31,RPA=RPA)
        phase = np.exp(2j*np.pi*q1*x) # plane wave on the two sites
        out.append([(phase.conj()@c[4:6,4:6]@phase).imag/2 for c in chi])
    return np.array(out).T
(s0,s) = (structure_factor(False),structure_factor(True))

def draw(axs, c):
    for (ax,z,title) in zip(axs,[s0,s],["free spinons","RPA"]):
        im = ax.pcolormesh(np.arange(len(qs)),energies,z,cmap=c.cmap,
                           vmin=0,vmax=2.,shading="nearest")
        ax.set_xticks([0,20,40,59],[r"$\Gamma$","K","M",r"$\Gamma$"])
        ax.set_title(title)
        ax.set_xlabel("momentum")
    style.colorbar(im, axs[1], r"Im $\chi^{zz}$ [$1/J$]")
    axs[0].set_ylabel("energy [$J$]")

style.save(draw, "pseudofermions_spinliquid", npanels=2)
i = np.argmax(s[:,20]) ; print(energies[i], s[i,20], s0.max(), s.max())
print(h2.get_rpa_stiffness([2/3,1/3,0.],nk=31,delta=1e-3,T=2e-3))
