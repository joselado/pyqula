# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


import numpy as np
import matplotlib.pyplot as plt
from pyqula import geometry

g = geometry.square_ribbon(3)
g = g.get_supercell(3)
g = g.remove(g.closest_index([0.,0.,0.]))

#g = g.get_supercell(3)
#plt.scatter(g.r[:,0],g.r[:,1])
#plt.show()
#exit()

h = g.get_hamiltonian(has_spin=False)


fig = plt.figure()
(k,e) = h.get_bands(write=False)
plt.scatter(k,e,c="black",s=20,label="original bands")
# Wannierize the isolated, 0.62 eV-gapped 4-band window [34,37]. Default
# (deterministic, not random) trial projection: the first num_wann=4
# columns of the identity matrix in the original orbital basis.
nk = 20
trial_vectors = []
trial_vectors = np.array(trial_vectors).T + 0.0
hwan = h.get_wannier_hamiltonian(bands=[0, 0], nk=nk,
                                 num_iter=1000)

print("Wannier centres (Cartesian):\n", hwan.wannier_centres)
print("Wannier spreads:", hwan.wannier_spreads)
print("Total spread Omega:", hwan.wannier_spread_total)

(kw,ew) = hwan.get_bands(write=False)


import matplotlib.pyplot as plt
plt.scatter(kw,ew,c="red",s=6,label="Wannierized valence band")
plt.xlabel("k-path")
plt.ylabel("Energy")
plt.legend()


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

fig, axes = plt.subplots(num_wann, 1, figsize=(10, 3.2 * num_wann), squeeze=False)
axes = axes[:, 0]
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
    ax.scatter(xs, ys, s=np.sqrt(sizes), c=ws, cmap="inferno", zorder=2, edgecolors="k", linewidths=0.3)
    ax.scatter([cx], [cy], marker="x", c="cyan", s=200, zorder=3, label="centre")
    ax.set_title(f"Wannier function {n}  (spread={hwan.wannier_spreads[n]:.2f} $\\AA^2$)")
    ax.set_xlim(-window, window)
    ax.set_ylim(-6, 6)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylabel("y (Å)")

axes[-1].set_xlabel("x (Å)")
plt.tight_layout()
plt.show()
