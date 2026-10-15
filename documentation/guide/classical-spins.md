# Classical spin models

Up to now every model in this guide has been quantum, a Hamiltonian of
fermionic operators to diagonalize. Let us now turn to classical spins, unit
vectors sitting on the sites of a lattice, with `classicalspin.SpinModel`. Each
spin is parametrized by a pair of angles $(\theta_i,\phi_i)$, and the spins
interact through a real-space tensor exchange $\vec S_i \cdot J_{ij} \cdot \vec S_j$
and an optional Zeeman field. Like the `LatticeGas` of the next chapter, the
model takes its lattice and neighbor shells from a `Geometry` but is otherwise
independent of the quantum `Hamiltonian`: the energy is evaluated directly from
the angles, and the ground state is found by minimizing it, a local
gradient-based minimization started several times from random angles, rather
than by diagonalization. Only the $\Gamma$ point is supported, meaning that the
spin configuration repeats with the unit cell you give it, so an incommensurate
texture such as a spiral needs an explicit supercell large enough to hold it.

Let us take the simplest frustrated case, a triangular lattice with an
antiferromagnetic first-neighbor exchange, where the three bonds of a triangle
cannot all be satisfied at once:

```python
from pyqula import geometry
from pyqula import classicalspin

g = geometry.triangular_lattice() # geometrically frustrated lattice
g = g.get_supercell(3)

sm = classicalspin.SpinModel(g) # classical spin model on this geometry
sm.add_heisenberg(Jij=[1.0]) # first-neighbor antiferromagnetic exchange
sm.minimize_energy(tries=10) # multistart local minimization

mx,my,mz = sm.get_magnetization() # per-site magnetization components
```

`mx`, `my` and `mz` are the components of the unit vector on each site. What
comes out is the 120-degree state: every pair of neighboring spins forms an
angle of 120 degrees (their scalar product is $-1/2$ on every bond), the three
spins of each triangle add up to zero, and so does the total magnetization, at
an energy of $-3J$ per site. The $3\times3$ supercell is what makes this
possible, since the pattern has three sublattices; on the primitive cell the
only $\Gamma$-point configuration is the ferromagnetic one. The `tries`
keyword is the number of random starting points of the minimization, and it
matters for a frustrated model, which has many local minima: the lowest energy
found over the restarts is the one kept.

`add_heisenberg` builds shell-based isotropic (or, via `Jm=[Jx,Jy,Jz]`,
diagonally anisotropic, XXZ) couplings the same way `g.get_hamiltonian(tij=...)`
does, so a list of exchanges is read as first, second, third neighbors. For
other forms of coupling, `classicalspin.generating_functions(name=...)` returns
ready-made two-point coupling functions: `"Linear"` (dipolar $1/r^3$),
`"RKKYTI"` (RKKY on a topological-insulator surface, PRB 81 233405), `"ZZ"` and
`"XYZ"` (Ising and anisotropic-diagonal), `"DM"` (Dzyaloshinskii-Moriya). These
go into `add_tensor(fun)` (couplings within the home cell) or
`add_tensor_2d(fun,ncells=...,vspiral=...)`, which also sums the periodic
images and can twist the coupling tensor by a per-image angle to embed a
spin-spiral wavevector. `get_local_energy()` gives the per-site energy, for a
spatially resolved map of a skyrmion or a domain wall, and
`classicalspintk.align.most_perp_basis()` rotates a texture into the frame
where it is mostly in-plane, which is the frame to draw it in with arrows.

See `examples/classicalspin/` for runnable versions, including the 120-degree
state above drawn as arrows together with its local energy (`perpendicular/`)
and a ladder with a modulated exchange whose texture is then fed into a quantum
Hamiltonian (`modulated_ladder/`). See
`jupyter-notebooks/functionalities/classical_spin_models_monte_carlo/01_classical_heisenberg.ipynb` and
`02_magnetization_textures.ipynb` in the same folder for executed notebooks.
