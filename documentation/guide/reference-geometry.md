# Geometry functions and methods

## g.get_hamiltonian()
Generate the Hamiltonian from a geometry.

Optional arguments

- tij = [1.0,.0,0.]: List with 1st, 2nd, 3rd nearest-neighbor hopping, or a
  function of two positions returning the hopping between them, or a
  `specialhopping.HoppingGenerator`
- has_spin=True: include the spin degree of freedom
- is_sparse=False: store the matrices in sparse form, for large cells
- non_hermitian=False: build a non-Hermitian Hamiltonian, which routes the
  band structure, DOS, LDOS and Berry curvature to their non-Hermitian
  implementations (see "Non-Hermitian Hamiltonians")
- is_multicell=False: store the hoppings as a multicell dictionary
- nc=2: neighbor cutoff used by the multicell construction

An unrecognized keyword raises `TypeError` listing the accepted ones, so a
misspelled `has_spin` is not silently the default.

Returns the Hamiltonian

## g.remove()
Return a copy of the geometry with sites removed. Used to carve a vacancy
or an antidot out of a flake or a supercell before building its
Hamiltonian (see "Valley operator" for an example).

Arguments

- i=0: which sites to drop, a single index into `g.r`, a list of indices,
  or a callable of the position, in which case every site where it returns
  True is removed

Returns a new geometry; `g` itself is untouched

## g.get_supercell()
Generate a supercell

Arguments

- nsuper: size of the supercell to create, number or tuple, or a 3x3 integer matrix M for a general non-diagonal/non-orthogonal supercell (see "Electronic structure folding and unfolding" for how this interacts with `operator="unfold"`) (positional, or by that name)

Optional arguments

- store_primal=False: keep a reference to the primitive-cell geometry on the supercell, needed by `operator="unfold"` (see "Electronic structure folding and unfolding")

A non-integer `nsuper` asks for a cell that many times larger by area, not that
many primitive cells along each vector, so that `np.sqrt(3)` gives the
$\sqrt3\times\sqrt3$ cell; the most compact cell of that size is returned, and
this route is for 2d geometries only

Returns a new geometry

## g.get_unfolded_kpath()
Return the k-path of the primitive cell, written in the reduced coordinates of
this supercell, which is the path an unfolded band structure has to be computed
along (see "Which k-path to unfold along"). Available on the Hamiltonian as
`h.get_unfolded_kpath()` as well, and it needs the supercell to have been built
with `store_primal=True`

Arguments

- kpath=None: the same as in `g.get_kpath()`, either an explicit list of
  k-points or a list of high-symmetry labels, which are read in the primitive
  cell and not in the supercell
- g0=None: the primitive-cell geometry, when the supercell does not carry one
- nk: number of k-points along the path, passed through to `g.get_kpath()`

Returns an array of k-points, one per row, ready to be passed as `kpath=` to
`h.get_bands()` or `h.get_kdos_bands()`
