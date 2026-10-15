# Setting up a Hamiltonian

Let us start with the simplest tight-binding model there is, a one-dimensional chain with
hopping between first neighbors, and see how its band structure is computed. Everything else
in this guide is built the same way: a geometry, the Hamiltonian generated from it, terms
added to that Hamiltonian, and a quantity computed from the result. In this chapter we
generate the Hamiltonian and add the first few terms to it, longer-range hoppings, an onsite
energy, a Zeeman field, an orbital magnetic field and a filling; the observables come in the
next chapter.

The Hamiltonian of a one-dimensional tight-binding chain takes the form

$$H = \sum_n c^\dagger_n c_{n+1} + h.c.$$

where $c^\dagger_n$ creates an electron at site $n$, so that each term moves an electron from
site $n+1$ to site $n$ and its hermitian conjugate moves it back. The hopping is the unit of
energy, $t=1$, and the distance between sites is the unit of length. This model can be
diagonalized analytically, giving rise to a diagonal Hamiltonian of the form

$$
H = \sum_k \epsilon_k \Psi^\dagger_k \Psi_k
$$

where the energy-momentum dispersion takes the form

$$
\epsilon_k = 2\cos{k}
$$

With the pyqula library, the previous band structure can be computed as

```python
from pyqula import geometry
g = geometry.chain() # geometry of the 1D chain
h = g.get_hamiltonian() # generate the Hamiltonian
(k,e) = h.get_bands() # compute band structure
```

The geometry holds the positions of the sites and the lattice vector, the Hamiltonian is
generated from it with first-neighbor hopping, and `h.get_bands()` returns the k-points along
the path and the energies at each of them, in this case the cosine band above. The
Hamiltonian is spinful by default, meaning that every energy appears twice, once per spin; a
spinless one is generated with `g.get_hamiltonian(has_spin=False)`. The methods that add a
term to a Hamiltonian, all of them named `add_something`, modify `h` in place, so a
calculation is a sequence of calls on the same object.

See `examples/1d/linear_chain/main.py` for a runnable version ending in a plot (with a
second-neighbor hopping added, so the band is not the pure cosine above), and the notebooks
in `jupyter-notebooks/functionalities/single_particle_hamiltonians/` for the spinless,
spinful and Nambu bases and for models from zero to three dimensions.

## Including second and third neighbor hopping

By default, the Hamiltonian generated includes only first-neighbor hopping, $t_1=1$. We will
now see how to include hopping to sites further away, considering a generalized Hamiltonian
of the form

$$
H = 
\sum_n c^\dagger_n c_{n+1} +
t_2\sum_n c^\dagger_n c_{n+2} +
t_3\sum_n c^\dagger_n c_{n+3} +
h.c.
$$

where $t_2$ and $t_3$ are the second- and third-neighbor hoppings, in units of $t_1$. Passing
the list of hoppings to `g.get_hamiltonian()`, first neighbor first, generates this
Hamiltonian; to compute the eigenvalues taking $t_2 =0.2$ and $t_3=0.3$, we write

```python
from pyqula import geometry
g = geometry.chain() # geometry of the 1D chain
h = g.get_hamiltonian(tij=[1.0,0.2,0.3]) # Hamiltonian with t1,t2,t3
(k,e) = h.get_bands() # compute band structure
```

The dispersion is now $\epsilon_k = 2\cos k + 2t_2\cos 2k + 2t_3\cos 3k$, and what you see
in the band structure is that the band is no longer symmetric between positive and negative
energies: the second-neighbor hopping is what breaks that symmetry, while the first- and
third-neighbor ones preserve it. The same list works for any geometry, and a function of the
two positions or a hopping generator can be passed instead of the list; see
`g.get_hamiltonian()` in the reference chapter.

See `examples/1d/NNN_chain/main.py` for a runnable version.

## Including an onsite energy

An onsite energy changes the energy of an electron that sits on a site, without moving it.
With the same value $\mu$ on every site it is a chemical potential,

$$
H =
\mu \sum_n c^\dagger_n c_{n}
$$

meaning that the whole band structure is shifted rigidly by $\mu$. Adding a uniform onsite
energy is therefore how the Fermi energy of a Hamiltonian is moved, since the observables
that depend on it, the filling, the density of states at the Fermi level, the expectation
values, take zero energy as the Fermi energy. This can be added to the Hamiltonian with
`h.add_onsite()` as

```python
from pyqula import geometry
g = geometry.chain() # geometry of the 1D chain
h = g.get_hamiltonian() # generate the Hamiltonian
mu = 0.3 # value of the onsite
h.add_onsite(mu) # add onsite energy
```

Possible inputs

- Float: the same onsite energy is added to all the sites

- Iterable (list or array): adds a different onsite energy to each site in the geometry, one
  value per site

- Callable (function): adds a different onsite energy to each site according to its location
  $\mathbf r$

A site-dependent onsite energy is how a potential landscape is built: a sublattice imbalance
in the honeycomb lattice, a single impurity (an onsite energy much larger than the bandwidth
on one site, as in the quasiparticle interference section), or a smooth electrostatic
potential given as a function of the position.

## Including an external Zeeman field

We will now add an external magnetic field acting on the spin of the electrons, a Zeeman
field, with `h.add_zeeman()`. We now include the existence of a spin degree of freedom,
considering the Hamiltonian

$$
H = H_0 +H_Z
$$

where $H_0$ is the original tight-binding Hamiltonian

$$
H_0 = \sum_{n,s} c^\dagger_{n,s} c_{n+1,s} + h.c.
$$

and

$$
H_Z = \sum_{n,s,s'} \vec B \cdot \vec \sigma^{s,s'} c^\dagger_{n,s} c_{n,s'}
$$

with $n$ running over the sites and $s,s'$ running over the spin degree of freedom. The
magnetic field takes the form $\vec B = (B_x,B_y,B_z)$, and $\sigma_\alpha$ are the spin
Pauli matrices, so that the field splits the two spin bands by $2|\vec B|$ and sets the
direction along which the spin is quantized. To add a magnetic field of the form
$\vec B = (0.1,0.2,0.3)$ to our chain we write

```python
from pyqula import geometry
g = geometry.chain() # geometry of the 1D chain
h = g.get_hamiltonian() # generate the Hamiltonian
h.add_zeeman([0.1,0.2,0.3]) # add the Zeeman field (modifies h in place)
(k,e) = h.get_bands() # compute band structure
```

The Hamiltonian is spinful by default, so the field acts on a spin degree of freedom that is
already there; on a spinless Hamiltonian this call adds the spin degree of freedom first.
The field can also differ from site to site, given as one vector per site or as a function
of the position. The same kind of term, written as the magnetization of the material rather
than as an external field, is `h.add_exchange()`: the two add the same matrix, and the name
follows the physics, an exchange field being what a magnet has and what we will find
self-consistently in the mean-field chapter.

## Including an external orbital field

A magnetic field also acts on the orbital motion of the electrons, through the phase that a
hopping picks up when an electron goes from one site to another. This is the Peierls
substitution

$$
t_{\alpha \beta} \rightarrow t_{\alpha \beta} e ^{i\int_{r_\alpha}^{r_\beta} \vec A \cdot d \vec l}
$$

where $\vec A$ is the vector potential, so that $\vec B = \nabla \times \vec A$, and the
integral runs along the bond. Let us apply it to a ribbon, a system finite in one direction
and periodic in the other, with `h.add_orbital_magnetic_field()`

```python
from pyqula import geometry
N = 20 # number of unit cells as the width
g = geometry.square_ribbon(N) # ribbon
h = g.get_hamiltonian() # generate the Hamiltonian
B = 0.02 # magnetic field in quantum flux unit
h.add_orbital_magnetic_field(B) # add an out-of plane magnetic field
(k,e) = h.get_bands() # compute the Landau-level band structure
```

`B` is the magnetic flux through a plaquette of unit area, in units of the flux quantum, so
that `B = 0.02` means one flux quantum every fifty plaquettes. What you see in the band
structure is the quantum Hall effect: flat bands, the Landau levels, the lowest of them close
to $-4 + 2\pi B$ for the square lattice, and between them dispersive bands that cross the
gaps, the chiral states at the two edges of the ribbon, propagating in opposite directions on
opposite edges. For a honeycomb ribbon the same call gives the Landau levels of the Dirac
equation, with a level pinned at zero energy.

See `examples/1d/landau_levels_zigzag_ribbon/main.py` for a runnable version on a honeycomb
ribbon.

## Setting a filling

Up to now the Fermi energy of the chain has been at zero energy, which for the cosine band is
half filling. If you want to enforce a certain filling $\nu$ in a Hamiltonian, the fraction
of the states that are occupied, so that
$$
\langle c^\dagger_n c_n \rangle = \nu
$$

on average over the sites and, in a spinful system, over the two spins, use

```python
from pyqula import geometry
g = geometry.chain() # chain
h = g.get_hamiltonian()
h.set_filling(0.7) # enforce a filling
```

`h.set_filling()` computes the Fermi energy that gives this filling on a k-mesh of `nk`
points per direction, and adds the onsite energy that brings it to zero, so that afterwards
every observable that takes zero as the Fermi energy is at the filling you asked for. The
filling is a single number, enforced on average over the whole system, with $\nu = 0.5$ being
half filling. For a metal the k-mesh matters, since the filling changes continuously with the
Fermi energy; for an insulator any Fermi energy inside the gap gives the same filling. The
same keyword goes into `h.get_mean_field_hamiltonian(filling=...)` in the mean-field chapter,
where the filling is kept fixed along the self-consistent calculation.

What if the filling should differ from site to site, for example to model a charge
imbalance between two sublattices imposed by a gate or a substrate? A single Fermi energy
cannot do that, so the way we do this is by giving `h.set_filling()` one filling per site,
and it then solves for one onsite energy per site such that every site reaches its own
filling

```python
import numpy as np
from pyqula import geometry
g = geometry.honeycomb_lattice() # honeycomb lattice, two sites per unit cell
h = g.get_hamiltonian() # first-neighbor Hamiltonian
h.set_filling(np.array([0.4,0.6])) # filling 0.4 on sublattice A, 0.6 on sublattice B
occ = h.get_vev() # occupation of each site
```

Each entry keeps the convention of the scalar filling, the fraction of the states of that
site that are occupied, so in this spinful example `occ` comes out as roughly 0.8 and 1.2
electrons, and the average filling is the mean of the array, here half filling. The same
per-site solve is reached with a scalar and `average=False`, which enforces the same filling
on every site rather than on average. Note that on a finite island the levels are discrete
and the total number of electrons is an integer, so fillings that add up to a fractional
number of electrons are only reached through the broadening $\delta=0.01$ the solver uses,
and the occupations should be read with `h.get_vev(delta=1e-2)`.
