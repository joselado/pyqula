# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Local density of states of every site of a large island, the map STM
# measures, from a Chebyshev expansion: the moments of each site come from
# the recursion on the sites within the light cone of the expansion, so the
# maps are those of the whole island while the time grows linearly with
# the number of sites.
import numpy as np
from pyqula import geometry

g = geometry.honeycomb_lattice().get_supercell(80) # 12800 sites
g.dimensionality = 0 # a finite island with zigzag edges
h = g.get_hamiltonian(is_sparse=True,has_spin=False) # keep it sparse
energies = np.linspace(-1.,1.,41)
(x,y,es,ldos) = h.get_multildos(energies=energies,mode="KPM",
        delta=0.1, # energy resolution, half width of the peak of a level
        write=False) # one map per energy, all from the same moments

# the sites at the zigzag edges have two neighbors instead of three
from scipy.sparse import csr_matrix
neighbors = np.array((abs(csr_matrix(h.intra))>0).sum(axis=1)).ravel()
edge = neighbors<3
for e in [-1.,0.,1.]:
    d = ldos[np.argmin(np.abs(es-e))] # the map at this energy
    print("E =",e,": weight at the edges",np.round(d[edge].sum()/d.sum(),3),
            "on",np.round(edge.mean(),3),"of the sites")

# summing a map over the sites gives the density of states, here against
# the stochastic trace of h.get_dos(mode="KPM")
(e2,dos) = h.get_dos(mode="KPM",energies=energies,delta=0.1,write=False)
print("Sum of the maps / KPM DOS at E=-1, 0, 1:",
        np.round(np.sum(ldos,axis=1)[[0,20,40]]/dos[[0,20,40]],3))

# a radius below the light cone keeps a finite cluster around each site,
# cheaper, and here converged to the precision of an STM map
(x,y,d40) = h.get_ldos(e=0.,mode="KPM",delta=0.1,kpm_radius=40,write=False)
print("kpm_radius=40, largest change of the zero-energy map:",
        np.max(np.abs(d40-ldos[20]))/np.max(ldos[20]))
