# Twisted bilayer graphene structural relaxation

In a twisted graphene bilayer the local stacking changes along the moiré
pattern, between AA regions, where the atoms of the two layers are on top of
each other, and AB and BA (Bernal) regions. AA stacking costs more energy
than AB and BA. So for small twist angles, below a few degrees, the real
lattice does not stay rigid: the atoms move in the plane, the AA regions
shrink, and triangular AB and BA domains grow, separated by domain walls.
This relaxation changes the band structure, so it should be included before
the electronic structure is computed. We do this with `GrapheneGeometry` and
its method `.relax()`.

## The energy that is minimized

`.relax()` finds the in-plane displacement $\mathbf u$ of each atom that
minimizes the sum of two energies, with the parameters of Carr, Massatt,
Torrisi, Cazeaux, Luskin and Kaxiras,
[arXiv:1805.06972](https://arxiv.org/abs/1805.06972).

The first one is the interlayer energy, the generalized stacking fault
energy. It depends on the local shift between the two layers, written as two
phases $(v,w)$, so that it is periodic in the shift:

$$
V(v,w) = c_0 + c_1\left[\cos v+\cos w+\cos(v+w)\right]
+ c_2\left[\cos(v+2w)+\cos(v-w)+\cos(2v+w)\right]
+ c_3\left[\cos 2v+\cos 2w+\cos(2v+2w)\right],
$$

per unit cell, with $c_0=6.832$, $c_1=4.064$, $c_2=-0.374$ and
$c_3=-0.095$ meV. AA stacking is its maximum, and AB and BA are its two
degenerate minima.

The second one is the elastic energy of each layer, which is the cost of
deforming it,

$$
E_{\rm el} = \frac{1}{2}\left[ G\,(\partial_x u_x+\partial_y u_y)^2
+ K\left( (\partial_x u_x-\partial_y u_y)^2 + (\partial_x u_y+\partial_y u_x)^2 \right)\right],
$$

per unit cell, with $G=69518$ meV and $K=47352$ meV. The first term is the
cost of a change of area, the second one the cost of a shear. The atoms only
move in the plane; corrugation out of the plane is not included.

The result is a balance between the two: the interlayer energy wants large
AB and BA domains, and the elastic energy wants a smooth displacement. For
small angles the moiré cell is large, so the domains can grow while the
deformation stays smooth, and the relaxation is strong.

## The electronic structure of the relaxed lattice

`GrapheneHamiltonian` builds the tight-binding Hamiltonian from a relaxed or
a rigid `GrapheneGeometry`. Its hopping between two sites depends on their
distance $r$ in three dimensions,

$$
t(\mathbf r) = -t\,\frac{r_{\parallel}^2}{r^2}\,e^{-\lambda(r-1)}\,e^{-\lambda_z r_z^2}
- t_\perp\,\frac{r_z^2}{r^2}\,e^{-\lambda_\perp(r-d)} ,
$$

where $r_\parallel$ and $r_z$ are the in-plane and out-of-plane parts of
$\mathbf r$. The first term is the in-plane hopping, $t=1$, and the second
one the interlayer hopping, $t_\perp=0.12$, with an interlayer distance
$d=3$ in units of the carbon-carbon distance. Note that the hopping here is
$-t$, the opposite sign of the $+1$ of `g.get_hamiltonian()`, so the bands
of the two Hamiltonians are mirrored in energy. Because the hoppings depend on
the real distances, the relaxed positions change the electronic structure
without any other change.

```python
from pyqula import specialgeometry
from pyqula.graphenetk.geometry import GrapheneGeometry
from pyqula.graphenetk.hamiltonian import GrapheneHamiltonian

g0 = specialgeometry.twisted_bilayer(m0=15) # ~2 degree twist
g = GrapheneGeometry(g0).relax() # AA area shrinks, AB/BA domains grow
h = GrapheneHamiltonian(g)
(k,e) = h.get_bands(num_bands=20)
```

`.relax()` returns a new geometry and does not change the rigid one, so you
can compare the two. `GrapheneGeometry` works for any graphene multilayer
(bilayer, twisted bilayer, twisted trilayer). The relaxation is stronger for
smaller twist angles. The band structure of the relaxed lattice is the one
to compare with an experiment on a real sample, for example with ARPES or
with the local density of states measured by STM.

See `examples/2d/graphene_relax/main.py` for a runnable script that compares
the rigid and the relaxed lattices, and
`jupyter-notebooks/functionalities/single_particle_hamiltonians/10_twisted_bilayer_relaxation.ipynb`
for an executed notebook.
