# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

import numpy as np
from pyqula import geometry
from pyqula.symmetrytk import pointgroup

# get_wannier_hamiltonian(..., symmetries=...) verifies point-group symmetries
# of the selected band group before Wannierizing, either auto-detected from
# the geometry+Hamiltonian ("auto") or supplied explicitly as a list of
# symmetrytk.pointgroup.SymmetryOperation. See wanniertk/wannierize.py's
# "symmetries" docstring for exactly what this does and does not guarantee:
# for a genuine (unitary) point-group symmetry the reconstructed
# Hamiltonian's spectral symmetry is already automatic once the selected
# bands are validated as a genuine union of symmetry-related multiplets --
# so this option mainly acts as that validation (rejecting a band range that
# slices through a symmetry-related degeneracy, e.g. a Dirac touching point)
# plus a numerical cleanup, not a correction.

g = geometry.kagome_lattice()
h = g.get_hamiltonian(has_spin=False)

# what point-group operations does this geometry+Hamiltonian actually have?
# (best-effort, dependency-free heuristic -- see pointgroup.find_point_group's
# docstring for what it can/can't find; kagome_lattice() has a site sitting
# exactly at the coordinate origin, so this finds the full C3v-about-that-site
# subgroup, on top of the origin's own C2h)
found = pointgroup.find_point_group(g, h=h)
print("Point-group operations found:", [c.op.name for c in found])

# kagome's famous flat band (index 2, the top band) is NOT, on its own, a
# valid symmetric band selection: it is exactly degenerate with the middle
# (dispersive) band at the K point, so no single band selection containing
# only it is a union of whole symmetry-related multiplets everywhere on the
# mesh -- get_wannier_hamiltonian raises instead of silently returning a
# mis-symmetrized model (this is the real, well-known topological
# obstruction behind why kagome's flat band alone has no symmetric
# exponentially-localized Wannier function -- not a bug in the check)
try:
    h.get_wannier_hamiltonian(bands=[2, 2], nk=12, num_iter=200, symmetries="auto")
    print("\n(unexpected: no error for the flat band alone)")
except ValueError as e:
    print("\nFlat band alone correctly rejected:\n ", str(e).splitlines()[0])

# the full 3-band manifold has no such obstruction (trivially a union of
# every multiplet) -- Wannierize it with the detected point group enforced
hwan_sym = h.get_wannier_hamiltonian(bands=[0, 2], nk=12, num_iter=500,
                                      symmetries="auto",
                                      trial_vectors=np.eye(3, dtype=complex))
print("\nSymmetries enforced:", [c.op.name for c in hwan_sym.wannier_symmetries])
print("Wannier centres (Cartesian):\n", hwan_sym.wannier_centres)
print("Total spread Omega:", hwan_sym.wannier_spread_total)

# compare bands: original vs Wannierized full manifold
(k, e) = h.get_bands(write=False)
(kw, ew) = hwan_sym.get_bands(write=False)

import matplotlib.pyplot as plt
plt.scatter(k, e, c="black", s=20, label="original bands")
plt.scatter(kw, ew, c="red", s=6, label="symmetry-verified full-manifold Wannierization")
plt.xlabel("k-path")
plt.ylabel("Energy")
plt.legend()
plt.show()
