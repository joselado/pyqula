# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

import numpy as np
import matplotlib.pyplot as plt
from pyqula import geometry

# Wannierize a 4-band composite group (bands=[34,37]) of a 3,7-triangulene
# ribbon -- the reproducer from GitHub issue #28 -- and plot the resulting
# real-space Wannier functions directly on top of the ribbon geometry, one
# panel per Wannier function, marker size/color proportional to the
# function's |amplitude|^2 on each site. See wannier_functions_from_gauge/
# WannierHamiltonian.wannier_functions for the underlying data.

def _geometry_bipartite(n, m, a1, a2, a3):
    R = [None] * (2 * n * m)
    ii = 0
    for i in range(n):
        for j in range(m):
            for k in range(2):
                R[ii] = i * np.array(a1) + j * np.array(a2) + k * np.array(a3)
                ii += 1
    return R

# build the 3,7-triangulene ribbon: a bipartite lattice with sublattice
# imbalance (hosts near-zero-energy "radical" bands), represented as a
# honeycomb zigzag ribbon whose site positions are overwritten by the
# triangulene-fragment tiling below
n1 = 3 * 7 * 2
n2 = 15
a1 = np.array([-np.sqrt(3) / 2, -3 / 2, 0])
a2 = np.array([np.sqrt(3), 0, 0])
a3 = np.array([-np.sqrt(3) / 2, -1 / 2, 0])

R1 = _geometry_bipartite(3, 7, a1, a2, a3)  # 3,7-triangulene fragment
for i in range(n1):
    if R1[i][0] > 5.25 * a2[0]:
        R1[i] = R1[i] - 7 * a2

R2 = np.array([np.array([0, 0, 0]), a3, a1, a1 + a3, 2 * a1, 2 * a1 + a3,
               a2 + a3, a1 + a2, a1 + a2 + a3, a1 + 2 * a2 + a3, 2 * a1 + a2,
               2 * a1 + a2 + a3, 2 * a1 + 2 * a2, 2 * a1 + 2 * a2 + a3,
               2 * a1 + 3 * a2 + a3])
R3 = -R2
for i in range(n2):
    R3[i] += (5 * a1 + 3 * a2 + a3)
    R2[i] += (-3 * a1 + 2 * a2)

N = n1 + 2 * n2
R = [None] * N
for i in range(n1):
    R[i] = R1[i]
for i in range(n1, n1 + n2):
    R[i] = R2[i - n1]
for i in range(n1 + n2, N):
    R[i] = R3[i - n1 - n2]
inter_vec = 7 * a2

R = np.array(R)
R[:, 1] -= np.mean(R[:, 1])  # center

g = geometry.honeycomb_zigzag_ribbon(10)
g.has_sublattice = False
g.r = R
g.a1 = inter_vec
g.r2xyz()

# fitted first/second/third-neighbor hoppings for this ribbon
t1 = -3.5228170498482223
t2 = 0.023303412692429096
t3 = -0.28680389683983526
h = g.get_hamiltonian(tij=[t1, t2, t3], has_spin=False)

# Wannierize the isolated, 0.62 eV-gapped 4-band window [34,37]. Default
# (deterministic, not random) trial projection: the first num_wann=4
# columns of the identity matrix in the original orbital basis.
nk = 40
hwan = h.get_wannier_hamiltonian(bands=[34, 37], nk=nk, num_iter=1000)

print("Wannier centres (Cartesian):\n", hwan.wannier_centres)
print("Wannier spreads:", hwan.wannier_spreads)
print("Total spread Omega:", hwan.wannier_spread_total)

# plot each Wannier function's real-space weight |amplitude|^2 on every
# site, translated across unit cells via wannier_functions' {R: (num_orbitals,
# num_wann)} real-space coefficients, zoomed to that function's own weighted
# centroid (each function can converge near a different reference cell when
# jointly Wannierizing multiple bands without disentanglement)
wf = hwan.wannier_functions
a1vec = h.geometry.a1
r0 = h.geometry.r
num_wann = hwan.wannier_spreads.shape[0]
Rcells = sorted(wf.keys(), key=lambda t: t[0])

fig, axes = plt.subplots(num_wann, 1, figsize=(10, 3.2 * num_wann))
window = 20  # Angstrom half-width shown around each function's centroid

for n in range(num_wann):
    ax = axes[n]
    xs, ys, ws = [], [], []
    for Rc in Rcells:
        shift = Rc[0] * a1vec
        coeff = wf[Rc][:, n]
        weight = np.abs(coeff) ** 2
        pos = r0 + shift[None, :]
        xs.append(pos[:, 0]); ys.append(pos[:, 1]); ws.append(weight)
    xs = np.concatenate(xs); ys = np.concatenate(ys); ws = np.concatenate(ws)
    cx = np.sum(xs * ws) / np.sum(ws)
    cy = np.sum(ys * ws) / np.sum(ws)

    for Rc in Rcells:
        pos = r0 + (Rc[0] * a1vec)[None, :]
        mask = np.abs(pos[:, 0] - cx) < window + 8
        ax.scatter(pos[mask, 0], pos[mask, 1], s=14, c="lightgray", zorder=1, edgecolors="none")

    sizes = 3000 * ws / ws.max()
    ax.scatter(xs, ys, s=sizes, c=ws, cmap="inferno", zorder=2, edgecolors="k", linewidths=0.3)
    ax.scatter([cx], [cy], marker="x", c="cyan", s=200, zorder=3, label="centre")
    ax.set_title(f"Wannier function {n}  (spread={hwan.wannier_spreads[n]:.2f} $\\AA^2$)")
    ax.set_xlim(cx - window, cx + window)
    ax.set_ylim(-6, 6)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylabel("y (Å)")

axes[-1].set_xlabel("x (Å)")
plt.tight_layout()
plt.show()
