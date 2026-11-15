# Geometry functions and methods

## g.get_hamiltonian()
Generate the tight-binding Hamiltonian of the geometry. By default it has
first-neighbor hopping $t=1$ and is spinful. See
[Building a Hamiltonian](hamiltonian.md).

Optional arguments

- tij=None: the hoppings. `None` is first-neighbor hopping $t=1$. A list
  gives the 1st, 2nd, 3rd... neighbor hoppings, for example `[1.0,0.2]`. It
  can also be a function of the two positions that returns the hopping
  between them, or a `specialhopping.HoppingGenerator`; these reach the
  distance `rcut`
- has_spin=True: include the spin degree of freedom
- is_sparse=False: store the matrices as sparse matrices, for large unit
  cells
- non_hermitian=False: build a non-Hermitian Hamiltonian (see
  [Non-Hermitian Hamiltonians](non-hermitian.md))
- is_multicell=False: include hoppings to unit cells beyond the first
  neighbor cells. A list, a function or a generator in `tij` is always
  stored this way, so that no hopping beyond the neighbor cells is lost
- rcut=None: the distance a function or a generator reaches. Every pair
  of sites farther than `rcut` has no hopping, so each shell of neighbors
  is kept or dropped whole. `None` is 5.0 in a periodic system and every
  pair in a finite one. A longer hopping needs a larger `rcut`. A list
  reaches its last shell, and `rcut` with a list raises `TypeError`
- nc=None: a box of cells instead, $|n_i|\le$ `nc` along each lattice
  vector (`cutoff=` is the same), where every pair of sites is kept unless
  `rcut` is also given

A misspelled keyword raises `TypeError` with the list of accepted ones.

Returns the Hamiltonian

## g.remove()
Return a copy of the geometry with some sites removed. Use it to make a
vacancy or an antidot in a flake or in a supercell, before building the
Hamiltonian (see [Operators](operators.md) for an example).

Arguments

- i=0: the sites to remove: one index of `g.r`, a list of indices, or a
  function of the position, which removes every site where it returns True

Returns a new geometry; `g` is not changed

## g.get_supercell()
Generate a supercell

Arguments

- nsuper: size of the supercell. A number or a tuple multiplies the lattice
  vectors. A 3x3 integer matrix $M$ gives a general supercell, with lattice
  vectors $\mathbf a'_i=\sum_j M_{ij}\mathbf a_j$ (see
  [Band unfolding](unfolding.md))

Optional arguments

- store_primal=False: keep the primitive geometry inside the supercell. It
  is needed to unfold the bands with `operator="unfold"` (see
  [Band unfolding](unfolding.md))

For a 2D geometry, a non-integer `nsuper` gives a supercell whose area is
`nsuper`$^2$ times the primitive one, so `np.sqrt(3)` gives the
$\sqrt3\times\sqrt3$ cell.

Returns a new geometry

## g.get_unfolded_kpath()
Return the k-path of the primitive cell, written in the reduced coordinates
of the supercell. This is the path along which an unfolded band structure
is computed (see [Band unfolding](unfolding.md)). The same method exists as
`h.get_unfolded_kpath()`. The supercell must be built with
`store_primal=True`.

Arguments

- kpath=None: as in `g.get_kpath()`, a list of k-points or a list of
  high-symmetry labels, both read in the primitive cell
- g0=None: the primitive geometry, if the supercell does not contain it
- nk: number of k-points along the path

Returns an array of k-points, one per row, to give as `kpath=` to
`h.get_bands()` or `h.get_kdos_bands()`
