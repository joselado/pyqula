# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

import numpy as np
import matplotlib.pyplot as plt

from pyqula import specialgeometry
from pyqula.graphenetk.geometry import GrapheneGeometry
from pyqula.graphenetk.relax import _layer_groups

# Structural relaxation of twisted bilayer graphene: minimizes the sum of
# the interlayer GSFE adhesion energy and the intralayer elastic energy
# (Carr, Massatt, Torrisi, Cazeaux, Luskin, Kaxiras, arXiv:1805.06972) over
# an in-plane displacement field. Below a few degrees of twist this
# shrinks the (energetically costly) AA-stacked regions and grows
# triangular AB/BA (Bernal) domains around them -- see Nam & Koshino,
# arXiv:1706.03908.
g0 = specialgeometry.twisted_bilayer(m0=15)  # ~2 degree twist
g = GrapheneGeometry(g0).relax(verbose=True)

r0 = np.array(g0.r)
r2 = np.array(g.r)
layer0 = _layer_groups(r0[:, 2])[0]  # plot only the bottom layer

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, r, title in zip(axes, (r0, r2), ("rigid", "relaxed")):
    ax.scatter(r[layer0, 0], r[layer0, 1], s=4)
    ax.set_title(title)
    ax.set_aspect("equal")
plt.tight_layout()
plt.show()
