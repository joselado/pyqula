# Twisted bilayer graphene structural relaxation

Let us now see how the structural relaxation of a twisted graphene bilayer is included before
its electronic structure is computed, with `GrapheneGeometry` and its `.relax()` method.
Rigidly twisting two graphene layers is only an approximation: below a few degrees of twist,
the real lattice relaxes so that the energetically costly AA-stacked regions shrink and
triangular AB/BA (Bernal) domains grow around them, separated by solitonic domain walls (Nam
and Koshino, [arXiv:1706.03908](https://arxiv.org/abs/1706.03908)). `GrapheneGeometry` wraps
any graphene multilayer `Geometry` (bilayer, twisted bilayer, twisted trilayer, ...) and adds
a `.relax()` method that reproduces this effect by minimizing a phenomenological energy over
an in-plane relaxation displacement field. That energy has two pieces: the interlayer
generalized stacking fault energy (GSFE), a closed-form periodic function of the local
interlayer registry fit to graphene's AA, AB and BA stacking energies, and the intralayer
linear-elastic energy, both taken from Carr, Massatt, Torrisi, Cazeaux, Luskin and Kaxiras,
[arXiv:1805.06972](https://arxiv.org/abs/1805.06972), Table 1. The minimization runs entirely
in-plane: out-of-plane corrugation is not included.

`GrapheneHamiltonian` builds the actual tight-binding Hamiltonian from a (relaxed or rigid)
`GrapheneGeometry`, defaulting to the same distance-decaying hoppings as
`specialhamiltonian.twisted_bilayer_graphene`; since those hoppings depend on the true 3D
interatomic distance, the relaxed positions feed into the electronic structure automatically

```python
from pyqula import specialgeometry
from pyqula.graphenetk.geometry import GrapheneGeometry
from pyqula.graphenetk.hamiltonian import GrapheneHamiltonian

g0 = specialgeometry.twisted_bilayer(m0=15) # ~2 degree twist
g = GrapheneGeometry(g0).relax() # AA area shrinks, AB/BA domains grow
h = GrapheneHamiltonian(g)
(k,e) = h.get_bands(num_bands=20)
```

`.relax()` returns a new geometry and leaves the rigid one untouched, so the two can be
compared side by side. AA is the maximum of the stacking-fault energy and AB/BA its degenerate
minima, so the AA regions shrink and the AB/BA domains grow, by an amount that increases as
the twist angle decreases, and the band structure computed from the relaxed positions is the
one to compare against a measurement on a real sample. See
`examples/2d/graphene_relax/main.py` for a runnable version comparing the rigid and relaxed
lattices, and
`jupyter-notebooks/functionalities/single_particle_hamiltonians/10_twisted_bilayer_relaxation.ipynb`
for an executed notebook.
