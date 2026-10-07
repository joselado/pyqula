# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Self-consistent mean field of a large island kept sparse: with the
# Hamiltonian built with is_sparse=True, integration="kpm" holds the
# interaction, the density matrix and the mean field as sparse matrices,
# so the memory grows linearly with the number of sites, while the time of
# an iteration grows as its square. Two cases: a Hubbard antiferromagnet
# whose moments the Rashba coupling cants out of the collinear state, and a
# superconductor from an attractive U in an in-plane field.
import numpy as np
from pyqula import geometry

# a honeycomb island with Hubbard U and Rashba spin-orbit coupling
g = geometry.honeycomb_lattice().get_supercell(12) # 288 sites
g.dimensionality = 0 # make it finite
h = g.get_hamiltonian(has_spin=True,is_sparse=True) # keep it sparse
h.add_rashba(0.2) # spin-orbit coupling, the moments are not collinear
# the orientation of the moments, which the Rashba coupling selects only
# weakly, relaxes slowly, so the loop stops at a change of 1e-3 per entry of
# the mean field, under a hundred iterations
hmf = h.get_mean_field_hamiltonian(U=3.0,filling=0.5,mf="random",
        integration="kpm",npol=150,mix=0.5,
        maxerror=1e-3) # the self-consistent Hamiltonian
m = np.array([hmf.extract(c) for c in ["mx","my","mz"]]).T # exchange field per site
print("Antiferromagnet: mean |exchange field| =",np.mean(np.linalg.norm(m,axis=1)))
print("                 sublattice-staggered field =",
        np.linalg.norm(np.mean(m*np.array(g.sublattice)[:,None],axis=0)))

# the same loop with the recursion of every orbital truncated to the sites
# within kpm_radius hops of it, which makes an iteration linear in the
# number of sites; the antiferromagnet is gapped, so the density matrix
# decays exponentially and the truncation converges with the radius
for radius in [4,8]:
    np.random.seed(1) # the same random guess for both radii
    hr = h.get_mean_field_hamiltonian(U=3.0,filling=0.5,mf="random",
            integration="kpm",npol=150,mix=0.5,maxerror=1e-3,
            kpm_radius=radius) # truncated recursion
    mr = np.array([hr.extract(c) for c in ["mx","my","mz"]]).T
    print("  kpm_radius =",radius,": mean |exchange field| =",
            np.mean(np.linalg.norm(mr,axis=1)))

# a triangular island with attractive U, Rashba coupling and in-plane field
g = geometry.triangular_lattice().get_supercell(12) # 144 sites
g.dimensionality = 0 # make it finite
h = g.get_hamiltonian(has_spin=True,is_sparse=True) # keep it sparse
h.add_rashba(0.2) # spin-orbit coupling
h.add_exchange([0.1,0.,0.]) # in-plane Zeeman field
h.setup_nambu_spinor() # electrons and holes
hsc = h.get_mean_field_hamiltonian(U=-3.0,filling=0.3,mf="swave",
        integration="kpm",npol=150,mix=0.5) # the self-consistent Hamiltonian
print("Superconductor: mean |s-wave pairing| =",np.mean(np.abs(hsc.extract("swave"))))
