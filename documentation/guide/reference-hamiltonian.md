# Hamiltonian functions and methods

## h.get_bands()
Compute band structure

Optional arguments:

- nk = 400: number of k-points
- operator: a single operator, or a list of operators, to compute expectation values for at each eigenstate
- kpath: an explicit k-path, either as reduced coordinates or as a list of
  high-symmetry labels (`"G"`, `"M"`, `"K"`, `"X"`, `"Y"`, and in three
  dimensions `"Z"`, `"R"`, `"A"`, `"B"`)
- ewindow: a function of the energy returning whether to keep that band, used
  to restrict the output to an energy window (e.g.
  `ewindow=lambda e: abs(e)<0.5`)

- eigmode="complex": non-Hermitian Hamiltonians only, which part of the
  complex eigenvalue is returned and written, `"complex"`, `"real"` or
  `"imag"` (see "Non-Hermitian Hamiltonians"). With `"complex"` the written
  `BANDS.OUT` carries `k`, `Re E`, `Im E` and then the operator columns

- biorthogonal=False: non-Hermitian Hamiltonians only, weigh each state by $\langle L|O|R\rangle/\langle L|R\rangle$ instead of by its right eigenvector (see "The two spectral functions of a non-Hermitian Hamiltonian")

- write=True: also write `BANDS.OUT`, and the k-path files `KPOINTS_BANDS.OUT` and `BANDLINES.OUT`; `False` writes none of them (see "Files written to the working directory")

Without `kpath` the path is $\Gamma$-M for a square-like 2D lattice,
$\Gamma$-K-M-K'-$\Gamma$ for a triangular-like one, and
$\Gamma$-X-M-$\Gamma$-R for a 3D one; the 3D path leaves the $k_3=0$
plane, so the band edges of a 3D crystal are reached.

Returns kpoint index and energies, plus one extra row per operator if `operator` is given

## h.get_kdos_bands()
Compute a k-resolved spectral function (band structure dressed with a projection operator, or an unfolded spectral function) along a k-path.

Optional arguments:

- kpath: k-point path (auto-generated if not given)

- operator=None: operator used to weight the spectral function, e.g. `"unfold"` (see "Electronic structure folding and unfolding")

- energies, delta, nk: frequency range, broadening, k-point density

- mode="ED": `"ED"`, `"green"` or `"KPM"` (equivalently `use_kpm=True`); all three return the spectral weight summed over the orbitals, and `operator="unfold"` works with each of them

- frand=None: generator of the random vectors the KPM stochastic trace
  draws. Only the KPM path uses them, so passing it without `mode="KPM"`
  raises `ValueError`

- biorthogonal=False: non-Hermitian Hamiltonians only, the Green's-function spectral function instead of the right-eigenvector one, and required by `mode="green"` (see "The two spectral functions of a non-Hermitian Hamiltonian")

- write=True: also write `KDOS_BANDS.OUT` and the k-path files; `False` only returns the arrays (see "Files written to the working directory")

Returns k-path fraction, energy and spectral weight



## h.get_dos()
Compute the density of states, normalized so that its integral over the
energies is the number of states per unit cell.

Optional arguments:

- energies: array with frequencies of the DOS

- delta=None: broadening of the DOS. Left as `None` it is chosen from the
  k-mesh, `5/nk` (so 0.05 at the default `nk=100`), which keeps the curve
  smooth as the mesh is refined

- mode="ED": `"ED"`, `"KPM"`, `"adaptive"`, `"Green"` or `"RG"` (see the
  "Density of states" section). Anything else raises, listing those five

- nk=100: k-points per direction, for `"ED"` and `"KPM"`; `"adaptive"`
  uses it only as a subdivision limit of its error-controlled quadrature,
  and `"Green"`/`"RG"` forward it to the self-energy's own k-sum (see the
  "Density of states" section)

- operator=None: operator the DOS is projected onto (a name, a matrix or an
  `Operator`)

For a non-Hermitian Hamiltonian only `mode="ED"` exists, and the extra
`eigmode` argument chooses which part of the complex eigenvalue the
broadening is centred on, while `biorthogonal=True` sums the Green's-function
spectral function instead of the right-eigenvector one. See "Non-Hermitian
Hamiltonians".

Return energies and DOS

## h.get_gap()
Return the indirect gap, i.e. the smallest energy difference between an
empty and an occupied state anywhere in the Brillouin zone (the two need
not sit at the same k-point). Obtained by numerically minimizing over k
rather than by scanning a fixed mesh, so a gap that closes at an
incommensurate k-point is not missed.

Optional arguments:

- ntries=1: repeat the minimization this many times and keep the smallest
  result. The search is deterministic, so every repetition returns the same
  number and raising this only costs time

Returns a single number, the gap. Zero (up to numerical noise) for a metal
or a Dirac semimetal

## h.get_bandwidth()
Return the bottom and top of the spectrum, `(emin,emax)`; note that this
is the pair of band edges, not their difference. Uses the same k-space
optimization as `h.get_gap()`, so the edges are the true extrema over the
Brillouin zone rather than the extrema of a k-mesh sample

## h.get_filling()
Return the fraction of states below zero energy, i.e. the filling measured
with the Fermi energy at $E=0$. Half filling gives 0.5. Use
`h.set_filling(nu)` to shift the onsite energy so that a target filling is
realized, and this method to check the result.

For a Hamiltonian with the electron-hole (Nambu) degree of freedom the
spectrum is particle-hole symmetric, so exactly half of it lies below zero
whatever the density is; there the filling is instead the weight that the
occupied BdG states put on the electron components, normalized by the two
electron states per site. At zero pairing this reproduces the normal-state
filling exactly.

Optional arguments:

- nk: k-point density used to sample the spectrum

## h.get_total_energy()
Return the total energy, i.e. the sum of the occupied single-particle
eigenvalues. For a mean-field Hamiltonian this is the band energy only,
`h.get_mean_field_hamiltonian(...,return_total_energy=True)` returns the
interacting total energy including the double-counting correction instead

Optional arguments:

- nk=10: k-point density of the Brillouin-zone sum

- fermi=0.0: energy below which states are counted as occupied. Not accepted
  for a Nambu Hamiltonian (raises `ValueError`): a nonzero `fermi` is not a
  rigid shift of a BdG spectrum, since the electron and hole blocks shift by
  $-\mu$ and $+\mu$. Shift the Hamiltonian instead, with
  `h.shift_fermi(-mu)`, and leave `fermi=0`

- mode="mesh": k-space sampling; `use_kpm=True` switches to a Chebyshev
  estimate for large systems

For a Nambu/BdG Hamiltonian this returns the *electronic* energy, i.e. the
energy of the physical electrons rather than of the doubled Nambu spectrum,
so at zero pairing it agrees with the normal-state answer for the same
model. `nbands=` is not implemented there (raises `NotImplementedError`).
Note that this is still the band energy: the mean-field total energy of a
paired state, with the double counting of both the normal and the pairing
mean field removed, is the one returned with `return_total_energy=True`

## h.get_density_matrix()
Return the full density matrix of the occupied states, as a dense matrix in
the same basis as `h.intra`. See "Interactions at the mean-field level" for
the k-resolved, hopping-resolved version the self-consistent loops use.

Its index convention is
$\mathrm{dm}_{ij}=\sum_\mathrm{occ}\psi_i^*\psi_j$, the *transpose* of the
usual $\rho_{ij}=\sum_\mathrm{occ}\psi_i\psi_j^*$. So an expectation value is
`np.trace(dm.T@A)`, not `np.trace(dm@A)`: the two agree for any real
operator (the density, $\sigma_x$, $\sigma_z$, a projector) and differ by a
sign for a purely imaginary one ($\sigma_y$, the valley operator, any
current operator $i[H,r]$). Use `h.get_vev(operator)` rather than
contracting the matrix yourself

## h.get_ipr()
Return the inverse participation ratio of every eigenstate, as
`(energies,ipr)`. A delocalized state in a system of $N$ sites gives
$\mathrm{IPR}\sim 1/N$ and a state localized on one site gives
$\mathrm{IPR}\sim 1$, so this is the usual diagnostic for Anderson
localization or for in-gap bound states. Finite (0d) systems only: it
raises `NotImplementedError` for a periodic Hamiltonian; for those use
the IPR operator instead (see "Inverse participation ratio operator")

## h.get_vev() / h.get_single_vev() / h.get_several_vev()
Ground-state expectation values of operators, evaluated by summing over the
occupied states.

- `h.get_vev(operator=...)` returns one real number **per site**: the
  site-resolved expectation value of `operator` (any name accepted by
  `h.get_operator`, e.g. `"sz"`), or the site occupation if `operator` is
  omitted. This is what produces a magnetization or charge-density map
- `h.get_single_vev(A)` returns the single number $\langle A \rangle$ for
  one operator `A`, summed over the whole system
- `h.get_several_vev([A,B,...])` does the same for a list of operators in
  one pass, sharing the diagonalization

An operator that acts with a different matrix at every k-point, `"unfold"`
above all, is applied inside the sum over the Brillouin zone rather than
contracted with the k-summed density matrix, so `h.get_vev(operator="unfold")`
gives the unfolded weight site by site

With the electron-hole (Nambu) degree of freedom the three methods count a
normal observable once, so that a BdG Hamiltonian with zero
pairing gives the same numbers as the normal one, and a pairing operator such
as `"spair"` gives the anomalous expectation value, site by site with
`h.get_vev("spair")`, whose sum is `h.get_single_vev("spair")`

```python
from pyqula import geometry
g = geometry.honeycomb_lattice() # primitive geometry
gs = g.get_supercell(2,store_primal=True) # supercell, keeping the primitive cell
h = gs.get_hamiltonian(has_spin=False)
h.add_onsite(lambda r: 100.0 if sum((r-gs.r[0])**2)<1e-2 else 0.0) # a vacancy
d = h.get_vev(operator="unfold") # unfolded weight on each site
```

Optional arguments:

- nk=30: k-point density of the Brillouin-zone sum

## h.get_magnetization()
Site-resolved magnetization, as an `(nsites,3)` array. Two different
quantities go by this name, and the `mode` argument picks between them:

- `mode="vev"` (the default) returns the physical moment: the per-site
  expectation value
  $(\langle S_x\rangle,\langle S_y\rangle,\langle S_z\rangle)$ over the
  occupied states, i.e. `get_vev("sx"/"sy"/"sz")`. Being a Brillouin-zone
  integral it needs a k-mesh. Pass `nk`, or rely on the mesh a
  self-consistent Hamiltonian remembers from its own loop. On a metal at
  $T=0$ the moment is quantized in steps of $2/n_k$, so use a fine enough
  mesh (or `mode="field"`) when comparing weakly polarized states
- `mode="field"` reads the magnetic *term written in the Hamiltonian*
  instead, i.e. the coefficients of $\sigma_{x,y,z}$ on each site. After a
  self-consistent calculation this is the mean-field exchange field: the
  natural order parameter of the loop, proportional, not equal, to the
  moment, and continuous where the moment is quantized. On a Hamiltonian
  whose field you put in by hand it hands that field straight back

Note the sign: `add_zeeman`/`add_exchange` add $+\vec h\cdot\vec\sigma$, so
the occupied states polarize *against* $\vec h$ and the moment comes out
antiparallel to the field you applied.

```python
from pyqula import geometry
h = geometry.chain().get_hamiltonian()
h.add_exchange([0.,0.,0.5])
h.get_magnetization(nk=40)         # the moment: [0,0,-0.15], against the field
h.get_magnetization(mode="field")  # [0,0,0.5], the field you put in
```

Optional arguments:

- mode="vev": `"vev"` or `"field"`, as above
- any further keyword (e.g. nk) is forwarded to `get_vev` in `"vev"` mode


## h.get_topological_invariant()
Return a topological invariant of the occupied bands: in one dimension the
$Z_2$ invariant of a time-reversal-symmetric superconductor (class DIII),
$\pm 1$, and the Berry (Zak) phase otherwise, the $Z_2$ invariant in two
dimensions when the
Hamiltonian is time-reversal symmetric, a BdG Hamiltonian included, and the
Chern number otherwise, and in three dimensions the strong and weak $Z_2$
indices `(nu0, (nu1, nu2, nu3))`, each 0 or 1, when it is time-reversal
symmetric and the Chern vector `(C1, C2, C3)` otherwise. A 0d Hamiltonian
has no Brillouin zone and raises `ValueError`.


## h.add_soc()
Add Kane-Mele intrinsic spin-orbit coupling

Arguments:

- t: value of the SOC (positional, or by that name)


## h.add_zeeman()
Add a Zeeman field to the Hamiltonian

Arguments:

- zeeman: value of the Zeeman field, as a number (assumes [0,0,Bz]), array or
  callable function (positional, or by that name)


## h.add_rashba()
Add Rashba spin-orbit coupling

Arguments:

- c: value of the Rashba SOC (positional, or by that name)


## h.add_onsite()

Add a local onsite energy

Arguments:

- fermi: value of the onsite energy, as a number, an array with one entry
  per site, or a callable of the position (positional, or by that name)


## h.add_sublattice_imbalance()
Add a staggered onsite energy, $+m$ on one sublattice and $-m$ on the
other: the mass term that opens a gap in a honeycomb lattice and turns it
into a boron-nitride-like semiconductor

Arguments:

- mass: the imbalance, as a number or as a callable of the position

The term is written with the geometry's sublattice index, so it needs one.
A geometry with no sublattice (a chain, a triangular or square lattice) and
a geometry with more than two (kagome, pyrochlore, whose index runs
$0,1,2,\ldots$ rather than $\pm 1$) both raise a `ValueError`. On a
bipartite lattice you can label one
yourself with `g = g.get_supercell(2)` followed by `g.get_sublattice()`;
otherwise write the profile explicitly with `h.add_onsite(f)`.


## h.add_antiferromagnetism()
Add a staggered exchange field: a Neel pattern on two sublattices, and the
120-degree frustrated pattern on more than two

Arguments:

- mass: the exchange field, as a number or as a callable of the position

Like `add_sublattice_imbalance` this needs a sublattice and raises a
`ValueError` when the geometry has none. Unlike it, more than two
sublattices are supported, through the frustrated pattern

## h.add_haldane()
Add a Haldane term: a complex second-neighbor hopping whose sign is set by
the chirality of the two-step path, which opens a gap and makes a
honeycomb lattice a Chern insulator (see "Chern number").

Arguments

- t: amplitude of the second-neighbor hopping, a number or a callable of the position

Needs a geometry with a sublattice. Breaks time-reversal symmetry

## h.add_modified_haldane() / h.add_antihaldane()
Same second-neighbor complex hopping, but with the sign also flipped
between the two sublattices, so the two valleys acquire opposite masses
and the total Chern number is zero, a valley-Hall rather than a Chern
insulator. `add_antihaldane` is a second name for the same method.

Arguments

- t: amplitude of the second-neighbor hopping

## h.add_kane_mele()
Add a Kane-Mele spin-orbit term: the Haldane hopping with opposite sign for
the two spin channels, so time-reversal symmetry is preserved and the model
is a quantum spin Hall insulator instead of a Chern insulator. Turns the
Hamiltonian spinful if it is not already.

Arguments

- t: amplitude of the spin-orbit second-neighbor hopping

Its spin Chern number is what `topology.spin_chern` computes

## h.add_anti_kane_mele()
The sublattice-staggered counterpart of `add_kane_mele`, in the same way
`add_modified_haldane` is the counterpart of `add_haldane`.

Arguments

- t: amplitude of the second-neighbor hopping

## h.add_kekule() / h.add_chiral_kekule()
Add a Kekule bond modulation: a period-tripling pattern on the
first-neighbor bonds of a honeycomb lattice, which folds the two Dirac
points onto $\Gamma$ and gaps them. `add_kekule(t)` takes the modulation
amplitude. `add_chiral_kekule(t1=...,t2=...)` adds the bond-direction-aware
chiral version, whose two amplitudes are the two independent complex
components of the modulation, optionally on an explicit `registry=` of
retained hexagon centers rather than the default one

## h.add_valley_exchange()
Add a valley-space exchange term
$\vec{v}\cdot(\tau_x,\tau_y,\tau_z)$, the valley-pseudospin analogue of
`add_exchange` for real spin (see "In-plane valley operators").

Arguments

- v = (vx,vy,vz): the valley field

Built from the same in-plane valley operators, so for a periodic
Hamiltonian it needs a Kekule-commensurate cell, a multiple-of-3
supercell of the primitive honeycomb cell, and raises `ValueError`
otherwise. A finite (0d) flake needs no such commensurability

## h.add_peierls() / h.add_orbital_magnetic_field()
Add an out-of-plane orbital magnetic field as a Peierls phase on every
hopping. `add_orbital_magnetic_field` is a second name for the same method.

Arguments

- mag_field: the field, in units of the flux quantum per unit area of the
  lattice; the Peierls phase on a bond is $2\pi B\,y\,dx$ in the Landau
  gauge. A periodic (commensurate) calculation needs the flux through the
  unit cell to be a rational multiple of the flux quantum, so build the
  supercell first and pick `mag_field` to match it

Optional arguments

- gauge="Landau": `"Landau"` or `"symmetric"`

Refuses a Hamiltonian that already carries a pairing amplitude
(`NotImplementedError`): a Cooper pair has charge 2e, so the anomalous term
has no single Peierls phase, and an orbital field in a superconductor means
vortices. Add the field to the normal-state Hamiltonian first, then
`h.turn_nambu()`/`h.add_swave()`

## h.add_inplane_bfield()
Add an in-plane magnetic field, as a Peierls phase built from the
out-of-plane coordinate. Meaningful for a multilayer or a system with
finite thickness, where an in-plane field still threads flux between the
layers.

Optional arguments

- b=0.0: field strength
- phi=0.0: in-plane direction of the field, in units of $\pi$

Only implemented up to two dimensions

## h.add_strain()
Modify the hoppings according to a strain field, which is how a
pseudo-magnetic field is produced in graphene.

Arguments

- sr: a callable of the position returning the local strain

Optional arguments

- mode="scalar": `"scalar"` rescales every hopping by `sr(r)` alone;
  `"directional"` also uses the bond direction, for a genuinely
  anisotropic strain

Turns the Hamiltonian multicell

## h.add_crystal_field()
Add a crystal field: an onsite potential built from the local atomic
environment, so that sites with fewer or more distant neighbors (an edge,
a vacancy, the two inequivalent stackings of a bilayer) sit at different
energies.

Arguments

- v: strength of the crystal field. The built-in potential is normalized
  and has its average removed, so `v` sets the spread of the onsite
  energies rather than their offset

Optional arguments

- rcut=6.0: distance cutoff of the neighbor sum

## h.generate_spin_spiral()
Rotate the Hamiltonian into a spin-spiral ansatz, i.e. impose a magnetic
texture whose quantization axis winds with a wavevector $q$. Used both as a
starting point for a spiral mean-field calculation and to scan the energy
of the spiral as a function of $q$ (see "Spin-spin exchange interactions").

Optional arguments

- vector=[0.,0.,1.]: axis the spins rotate about
- qspiral=[1.,0.,0.]: the spiral wavevector
- fractional=True: read `qspiral` in fractional (reduced) coordinates. With
  `False` only the inter-cell hoppings are rotated

Needs a spinful Hamiltonian

## h.add_pairing()
Add a general superconducting pairing term, given by its symmetry channel,
turning the Hamiltonian into its Nambu (BdG) form if it is not already (see
"Spin-triplet d-vector and non-unitary superconductivity").

Optional arguments

- delta=0.0: pairing amplitude, a number or a callable of the position
- mode="swave": the pairing symmetry: `"swave"`, `"extended_swave"`,
  `"triplet"`, `"pwave"`, `"chiral_pwave"`, `"dx2y2"`, `"dxy"`,
  `"nodal_dwave"`, `"chiral_dwave"`, `"nodal_fwave"`, `"chiral_fwave"`,
  `"chiral_gwave"`, and several others. A
  callable returning the 2x2 pairing matrix for a pair of positions is also
  accepted, and it has to obey Fermi antisymmetry,
  $D(\vec r_1,\vec r_2)=\sigma_y D(\vec r_2,\vec r_1)^T\sigma_y$ (the singlet
  part even under the exchange of the two positions, the d-vector odd), or it
  raises `ValueError` naming the first pair that breaks it. An unknown name
  raises `ValueError` listing every accepted one
- d=[0.,0.,1.]: the d-vector, for the triplet channels, a list or a callable
  of the position

A `delta` or `d` given as a function of the position has to be periodic with
the lattice, since a bond between two cells is the same bond seen from either
of its ends and has to carry the same pairing from both; a function that is
not periodic raises `ValueError` naming the bond. A modulated pairing is set
up on a supercell commensurate with the modulation, and a zero-dimensional
geometry takes any function.

## h.setup_nambu_spinor()
Put the Hamiltonian into its Nambu (electron-hole doubled) form, with zero
pairing. This is the call the error messages of the superconducting
routines point at: a superconducting quantity asked of a normal Hamiltonian
needs the electron-hole degree of freedom to exist first. Equivalent to
`h.add_swave(0.0)`. A Nambu Hamiltonian is always spinful, so a spinless
one is made spinful first. Modifies in place

## h.turn_nambu()
Add the electron-hole degree of freedom without adding any pairing, making a
spinless Hamiltonian spinful first. Lower level than `setup_nambu_spinor()`,
which is the one to call normally. Modifies in place

## h.turn_spinful()
Add the spin degree of freedom to a spinless Hamiltonian, doubling every
matrix with an identity in spin space. This is the call the error messages
of the spin-resolved routines point at. Modifies in place, and does nothing
if the Hamiltonian is already spinful

## h.remove_spin()
The inverse: keep one spin block and discard the other, returning to a
spinless Hamiltonian.

Optional arguments

- channel="up": which block to keep, `"up"` or `"dn"`

This drops the spin off-diagonal block, so it is only meaningful when
spin is a good quantum number. With spin-orbit coupling or non-collinear
order the result is a different model, not a projection of this one. The
spin-splitting routines built on it check that explicitly (see
"Errors and unsupported inputs")

## h.turn_multicell()
Convert the Hamiltonian to multicell form, i.e. store the inter-cell
hoppings as a dictionary keyed by lattice vector instead of the single
`inter` matrix of the nearest-neighbor-cell form. Several routines require
it, and their error messages say so; hoppings beyond the first neighboring
cell need it. Modifies in place. `h.get_no_multicell()` returns a
non-multicell copy where that is possible

## h.get_hk_gen()
Return the Bloch generator: a function of the reduced k-vector returning
the Bloch matrix $H(k)$ of the Hamiltonian. This is the object every
k-space routine in the library is built on, and the way to evaluate a
model at a chosen k-point directly (`h.get_hk_gen()([0.3,0.,0.])`).
pyqula's convention is the periodic one, $H(k)=\sum_R t(R)e^{2\pi i k\cdot R}$
with $k$ in reduced coordinates and no intra-cell atomic positions in the
phases

## h.get_gf()
Return the bulk Green's function of the Hamiltonian at one energy, as a
matrix in the same basis as `h.intra`, obtained by integrating over the
Brillouin zone.

Optional arguments

- energy=0.0: energy at which it is evaluated
- delta=1e-5: imaginary part (analytic continuation)
- mode="adaptive": how the Brillouin-zone integral is done, `"adaptive"`
  (error-controlled), `"full"` (a fixed `nk` mesh) or `"renormalization"`
- gtype="bulk": `"bulk"` or `"surface"`

$-\mathrm{Im}\,\mathrm{Tr}\,G/\pi$ is the density of states, which is what
`h.get_dos(mode="Green")` computes from it

## h.get_hopping_dict()
Return the real-space hoppings as a dictionary `{(n1,n2,n3): matrix}`,
keyed by the lattice vector connecting the two cells, with `(0,0,0)` the
intracell block. This is the representation the response functions accept
for an interaction with support beyond one cell (see "Interactions beyond
onsite"), and the natural way to inspect or modify a Hamiltonian's hoppings
directly. `h.get_multihopping()` returns the same thing wrapped in a
`multihopping.MultiHopping`, which supports addition and scalar
multiplication, and `h.set_multihopping()` writes one back

## h.get_rkky()
Compute the RKKY interaction between two magnetic impurities mediated by
the electrons of `h`, i.e. the effective exchange coupling as a function of
their separation.

Optional arguments

- mode="pm": `"pm"` computes it explicitly ("poor man's"), by adding two
  local exchange fields to the Hamiltonian and taking the energy difference
  between their parallel and antiparallel alignment; `"LR"` computes it from
  linear response instead, which is much faster for a map over many
  separations. An unknown mode raises `ValueError` listing both
- for `mode="pm"`: `ri`, `rj` the positions of the two impurities (both
  required), `nk=10`, `dj=1e-1` the strength of the probe exchange fields
- for `mode="LR"`: `R=[0,0,0]` the lattice vector between the two cells,
  `ii=0`, `jj=0` the sites inside them, `nk=100`, `delta` the analytic
  continuation (default `1/nk`)

`rkky.rkky_map(h,n=...)` sweeps either mode over a range of separations.
On a bipartite lattice at half filling the sign is ferromagnetic between
sites of the same sublattice and antiferromagnetic between opposite ones

## h.get_ldos()
Compute the local density of states.

Optional arguments:

- e: energy of the LDOS (`energy` is accepted as an alias, since that is
  how the Green's function, embedding and transport routines spell it)

- delta=None: broadening of the LDOS, which must be positive, 0.001 when not
  given; in `mode="KPM"` the energy resolution of the expansion, with
  `npol=200` polynomials when neither is given

- operator=None: operator the LDOS is projected onto (a name, a matrix or an
  `Operator`); see the LDOS section above for how the modes weight it

- mode="arpack": `"arpack"` (diagonalization on a k-mesh), `"green"`
  (Green's function, 2D only) or `"KPM"` (Chebyshev expansion of every site,
  linear in the number of sites, see "The local density of states of a large
  system")

- npol, kpm_radius, scale, kpm_prec: `mode="KPM"` only, the number of
  polynomials, the radius in hops of the recursion of each site (`None` for
  the exact map), the window of the spectrum and the precision

- projection="TB": `"TB"`, `"TBRS"` (real-space interpolated) or `"atomic"`

- eigmode="complex": non-Hermitian Hamiltonians only, whether `e` is read
  on the real or the imaginary axis of the complex spectrum (see
  "Non-Hermitian Hamiltonians"). Only `mode="diagonalization"` is
  implemented there

Return x, position, y position and LDOS

## h.get_multildos()
Compute the LDOS at many energies. Returns `(x,y,energies,ldos)`, with
`ldos` one row per energy, and writes one file per energy to a
`MULTILDOS/` folder, together with a `MULTILDOS/DOS.OUT` holding the total
DOS on the same energies and a `DOSMAP.OUT` in the working directory,
unless `write=False`.
The maps and that DOS carry the same normalization as `h.get_ldos()` and
`h.get_dos()`, so they can be read against each other directly.

Optional arguments:

- energies=linspace(-1,1,100): energies to compute

- delta, nk: broadening and k-point density, as in `h.get_ldos()`, with
  `delta=0.01` when not given in the diagonalization mode

- mode="diagonalization": `"diagonalization"` or `"KPM"`, the Chebyshev
  expansion of every site, every energy from the same moments, taking the
  options of `h.get_ldos()` in that mode

- operator=None: operator the LDOS is projected onto, weighting each
  eigenstate by $\langle\Psi|A|\Psi\rangle$, the same convention
  `h.get_ldos(mode="arpack")` uses; in `mode="KPM"` the local matrix element,
  as in `h.get_ldos(mode="green")`. The older spelling `op=` is still
  accepted; passing both raises `TypeError`

- projection="TB": `"TB"` or `"atomic"`. Anything else raises `ValueError`
  listing the two, and `"atomic"` together with `operator=` raises
  `NotImplementedError` naming `"TB"` as the projection that supports it

An unrecognized keyword raises `TypeError`.

## h.get_chi()
Compute a non-interacting operator-operator response function (charge-charge by default).

Optional arguments:

- q=None: momentum transfer. Left as `None` the response is *averaged over
  the whole q-mesh* rather than evaluated at q=0; pass `q=[0.,0.,0.]`
  explicitly for the uniform response

- A=None, B=None: operators defining the response (default: identity, i.e. charge-charge)

- energies, delta, nk: frequency range, broadening, k-mesh density

Returns energies and the response function

## h.get_spinchi_ladder()
Compute the transverse ($S^+/S^-$) spin susceptibility, RPA-dressed by default using the Hubbard `U` of a mean-field Hamiltonian.

Optional arguments:

- q=None, energies, delta, nk: as above; with `q` left out the response is dressed at every $q$ of the mesh and averaged afterwards, the local RPA response (the bare one, `RPA=False`, is the average of `get_chi`)

- RPA=True: dress with the random-phase approximation; `False` for the bare response. An interaction that couples different sites is dressed in the pair basis of `h.get_transverse_spinchi()` instead of with the site vertex, with the same arguments (see "Interactions beyond onsite")

- T=None: temperature of the occupations, equal to `delta` when not given

- chi_prec=None: precision of the Lindhard sum, `"single"` or `"double"`; unset, it is `"single"` on the GPU and `"double"` on the CPU. Also accepted by `get_spinchi_full`, `get_qdos_iets`, `get_iets_ldos`, `get_rpa_kernel_poles` and `get_magnon_bands(method="rpa")`, and by `get_transverse_spinchi` and `get_magnon_bands(method="pair")`, which sum the response in the pair basis; `mode="trace"`/`"diagonal"` are double precision only, as is the pair basis on the CPU, and raise for `"single"`. Where the response is computed is the package-wide switch `gpu.set_gpu` (see "Running on a GPU"), and `mode="trace"`/`"diagonal"` and `imode="adaptive"` are not available on the GPU and raise

## h.get_rpa_kernel_poles()
Compute the poles of the generic RPA kernel $1-V(q)\chi(q,\omega)$: the frequencies of the collective modes/instabilities of the interacting response.

Optional arguments:

- V=None (required): the interaction; a `ValueError` is raised if not given. Either a plain matrix (q-independent, onsite-only) or a real-space hopping dict/`MultiHopping` `{(n1,n2,n3): matrix}` for an interaction with support beyond the onsite cell, Fourier-transformed to $V(q)$ at this call's `q` (see "Interactions beyond onsite")

- A=None, B=None, q=None, energies, delta, nk: as in `get_chi`; a pole belongs to one $q$, so with `q` left out the kernel is built from the q-averaged bare response at $q=0$, unlike the response functions above, which dress each $q$ before averaging

Returns an `(npoles,2)` array, one row per collective mode found, sorted by frequency: the pole frequency and its signed residual imaginary part (filter on its magnitude to keep only the sharp modes).

## h.get_magnon_bands()
Compute the magnon bands of a magnetic mean-field state, scanned along a q-path. Three methods, with different domains of validity:

Optional arguments:

- method="rpa" / "pair" / "tdhf": which ladder to sum. See "The three magnon routes" above for the coverage table; in short, "rpa" is the site basis for an onsite U and the pair basis for anything that couples different sites, "pair" keeps the interaction's pair index (any density-density interaction, onsite or not, and exchange through the spin channels the SCF records, metals included, frequency-resolved), "tdhf" solves the electron-hole pair eigenproblem (the same interactions, no frequency grid).

- `"rpa"` takes the poles of the full spin RPA kernel (the same $S_x,S_y,S_z$ channel as `get_spinchi_full`/`get_iets_ldos`), with the interaction taken from the mean field: the site vertex for an onsite `h.V`, and the pair-basis ladder of `"pair"`, with these same defaults, for a density-density or exchange interaction that couples different sites. It works for metals as well as insulators, and needs a frequency grid. `"tdhf"` solves the time-dependent Hartree-Fock problem in the spin-flip electron-hole pair basis: it handles a neighbor-shell density-density interaction and an exchange one whose spin channels the SCF recorded, has an exact Goldstone mode, needs no frequency grid, and requires a gapped reference converged on the same `nk` (or `metal=True`)

- qpath=None, nq=20: the q-path (default path of the geometry) and number of q-points

- energies, delta, nk: as above (`method="rpa"` only)

- nk, n, channel, V: for `method="tdhf"`, the k-mesh (which must match the SCF's), how many branches to keep per q-point, whether to restrict to the spin-flip block (`"auto"`, `"spinflip"`, `"all"`), and an interaction overriding `h.V`

Returns `(qs,ws,gammas)` for `method="rpa"`: three flat 1D arrays of equal length, `qs` the integer q-point index along the path, `ws` the pole frequency, `gammas` its residual imaginary part. `method="tdhf"` returns `(qs,es)`, with `es` the (complex) magnon energy.

## h.get_transverse_spinchi()
Return the spin response computed in the basis of the interaction's *pair* index rather than of sites, which is what lets it carry a neighbor-shell density-density interaction, the one the site-basis RPA maps to exactly zero, and an exchange interaction with its transverse part. Needs no gapped reference and no global spin quantization axis, so it covers metals and non-collinear states alike, and returns a frequency-resolved, spin- and site-resolved $\chi(\omega)$.

Optional arguments:

- W=None: the interaction, defaulting to the one the mean field was converged with

- q=None, energies, delta, nk: as in `get_chi`; with `q` left out each $q$ is dressed on its own and the average taken afterwards, the local RPA response

- component=None: a pair of spin indices `(a,b)` to return only that spin block instead of the full tensor

- chi_prec=None: precision of the Lindhard sum in the pair basis, as in `get_chi`; the CPU kernel of this route is double precision only and raises for `"single"`

Returns `(energies,chi)` with `chi` one $3N\times3N$ tensor per frequency, in the same $(S_x,S_y,S_z)\times$site layout `get_spinchi_full` uses.

## h.get_magnon_energies()
Return the magnon energies at a single center-of-mass momentum `Q`, from the spin-flip channel of the Bethe-Salpeter equation. Same arguments as `get_magnon_bands(method="tdhf")` with `Q=[qx,qy,qz]` in place of the q-path. A sizable imaginary part on an energy means the mean-field reference is unstable against that excitation.

## h.get_goldstone_residual()
Return how far a magnetic mean field is from having a zero-energy magnon at $Q=0$, as the Goldstone theorem requires of any magnetic state without spin-orbit coupling: $\|Mv\|/\|v\|$ with $M$ the time-dependent Hartree-Fock matrix and $v$ the uniform spin-rotation generator in the pair basis. It is proportional to the SCF tolerance the mean field was converged to and to nothing else, so it is the check to run before trusting a magnon dispersion. In particular it is what catches a mean field converged on a different `nk` than the magnon is being solved on.

Optional arguments:

- nk=10, V=None, channel="auto": as in `get_magnon_energies`

- relative=True: divide by the largest transition energy of the pair basis, so the number is comparable across models with different bandwidths

## h.get_densitychi_RPA()
Compute the density (charge) RPA response function for a `V1`/`V2`/`V3`-neighbor-shell (+ onsite `U`, + general `Vr(r)`) density-density interaction, same convention as `Vinteraction`/`VJinteraction`. Unlike `get_spinchi_full`, the interaction is taken directly as parameters, not read from `h.V`. No mean-field convergence is needed first.

Optional arguments:

- V1=0.0, V2=0.0, V3=0.0, U=0.0, Vr=None: the density-density interaction, built the same way as `Vinteraction`/`VJinteraction`'s
- rcut=None: range of `Vr`, every pair of sites up to that distance and none beyond it, the pair of a site with itself included; `None` means 5 in a periodic system and every pair in a finite one

- q=None, energies, delta, nk: as in `get_chi`; with `q` left out each $q$ is dressed with its own $V(q)$ and the average taken afterwards, the local RPA response

## h.get_plasmon_bands()
Compute the plasmon/charge-order bands: the poles of the density RPA kernel for a `V1`/`V2`/`V3`/`U`/`Vr` neighbor-shell density-density interaction, scanned along a q-path, the charge-channel analog of `get_magnon_bands`.

Optional arguments:

- V1=0.0, V2=0.0, V3=0.0, U=0.0, Vr=None: as in `get_densitychi_RPA`

- qpath=None, nq=20, energies, delta, nk: as in `get_magnon_bands`

Returns `(qs,ws,gammas)`, same convention as `get_magnon_bands`.

## h.get_bse()
Solve the Bethe-Salpeter equation (excitons) on top of this mean-field Hamiltonian, and return the solved `BSE` object.

Optional arguments:

- V=None: the electron-hole interaction. `None` reads it from `h.V`, the interaction the mean field was converged with (so the BSE kernel and the Fock self-energy inside `h` come from the same interaction). Otherwise a real-space dictionary `{(n1,n2,n3): matrix}`, or a plain matrix for an onsite-only interaction, `bsetk.interaction.density_interaction` builds one from `U`/`V1`/`V2`/`V3`/`Vr`

- Q=None: center-of-mass momentum of the exciton, defaulting to the zone center

- nk=10: k-points per direction of the mesh the electron-hole pairs are built on

- nv=None, nc=None: restrict to the `nv` highest valence and `nc` lowest conduction bands; `None` takes all of them

- kernel="full": which kernel terms to include, `"full"`, `"direct"` (ladder only), `"exchange"` (exactly the RPA) or `"none"` (bare transitions)

- tda=False: apply the Tamm-Dancoff approximation, diagonalizing only the resonant block

- max_memory=2.0: refuse, rather than attempt, a calculation whose dense matrix would need more than this many GB

- screening=None: the interaction of the *direct* (ladder) term. `None` keeps the bare one, i.e. time-dependent Hartree-Fock; `"rpa"` replaces it by the static RPA screened interaction $W=\varepsilon^{-1}v$ built from this Hamiltonian's own bands; `"crpa"` does the same with the transitions inside the `nv`/`nc` window left out of the polarization; a `ScreenedInteraction` reuses a precomputed one. The exchange term always keeps the bare interaction. Do not screen a fitted Hubbard `U`, see the section above

- nkW=None: k-mesh for the screening, defaulting to `nk`. Must be an integer multiple of it

- channel="charge": where the dielectric matrix is built. `"charge"` builds it on site indices (the standard GW construction, spin-rotation invariant); `"orbital"` dresses the full spin-orbital matrix as $\varepsilon^{-1}v$, which does not preserve SU(2). The two coincide for a spinless Hamiltonian

- solver="dense": how the eigenproblem is solved. `"dense"` builds the matrix and diagonalizes it, and is the only route to the full non-Tamm-Dancoff spectrum. `"iterative"` applies the exactly factorized kernel matrix-free and runs a preconditioned block LOBPCG, removing the memory wall. `"qtt"` compresses the kernel into a quantics matrix product operator and solves it by DMRG, with a cost growing like $\log N_k$. The last two need `tda=True`; see "Large k-meshes" above

- neig=None: how many excitons `solver="iterative"` returns; `"dense"` returns all of them and ignores it, and `"qtt"` accepts only `neig=1`. `None` means 4 for `"iterative"` and 1 for `"qtt"`

- gauge="auto": smooth the arbitrary phase left on each Bloch eigenvector. `"auto"` turns it on (as `"projection"`) only for `solver="qtt"`, which cannot work without it; `"phase"`, `"projection"` or `None` apply to every solver. It changes no energy, since it rotates the states only inside a degenerate multiplet, where every band has the same energy, and a non-degenerate band only by a phase

`solver="qtt"` additionally takes `tolerance` (default `1e-6`), `maxbonddim` and `maxdim`/`nsweep`/`cutoff`, which trade accuracy against cost, plus `coarse_nk` (the submesh the band window is read from) and `unfolding`.

The returned object exposes `energies`, `amplitudes` (the resonant amplitudes $A_{vc}(k)$), `amplitudesY` (the antiresonant ones, zero under `tda`), `pairs` (the k-mesh, band window and `(ik,iv,ic)` label of every pair index), and the methods `get_energies(n)`, `get_binding_energies(n)` and `get_lowest_transition()`. The last is the lowest independent-particle transition, the energy the binding energies are measured from.

## h.get_exciton_energies()
Return the exciton energies from the Bethe-Salpeter equation, sorted. Takes the same arguments as `get_bse`, plus `n=None` to keep only the `n` lowest.

## h.get_exciton_binding_energies()
Return the exciton binding energies, i.e. how far below the lowest independent-particle transition each exciton lies; positive means bound. Same arguments as `get_exciton_energies`.

## h.get_exciton_states()
Return `(energies,amplitudes)` of the excitons, the amplitudes being the electron-hole amplitudes $A_{vc}(k)$ of each one, indexed by the flattened pair index whose `(ik,iv,ic)` meaning is in the `BSE` object's `pairs.labels`. Same arguments as `get_exciton_energies`.

## h.get_exciton_bands()
Return the exciton band structure $E_X(Q)$: one Bethe-Salpeter solve per q-point along a path.

Optional arguments:

- qpath=None, nq=20: the q-path, same input as `get_bands` (a list of high-symmetry labels, a list of explicit q-vectors, or `None` for the default path with `nq` points)

- n=None: keep only the `n` lowest excitons at each q-point

- V, nk, nv, nc, kernel, tda, max_memory, screening, nkW: as in `get_bse`, passed through unchanged

Returns `(qs,es)`, flat 1D arrays of equal length: `qs` the integer index of the q-point along the path (same convention as `get_bands`) and `es` the exciton energy, complex at any q-point where the mean-field reference is unstable against the excitation. Note that `nv`/`nc` must not split a degenerate multiplet (a spinful Hamiltonian with no spin-orbit coupling or magnetic order needs an even `nv`/`nc`); a warning is issued if they do.

## h.get_screened_interaction()
Return the static RPA screened interaction $W(q)=\varepsilon^{-1}(q)v(q)$ built from this Hamiltonian's own bands, as a `ScreenedInteraction`.

Optional arguments:

- V=None: the *bare* interaction to screen, same forms as `get_bse`'s `V`. `None` reads `h.V`, but note that a fitted Hubbard `U` is already an effective screened interaction and should not be screened again

- nk=10: k-mesh for both the Brillouin zone sum and the q-grid `W` is tabulated on

- screening="rpa": `"rpa"` (all transitions polarize) or `"crpa"` (those inside `exclude` do not)

- exclude=None: `(vbands,cbands)` to leave out of the polarization, required by `"crpa"`

- channel="charge": `"charge"` (standard GW, on site indices, spin-rotation invariant) or `"orbital"` (the full spin-orbital matrix, which breaks SU(2))

- chi_prec=None: precision of the Brillouin zone sum that builds $\chi_0$, `"single"` or `"double"`; unset, it is `"single"` on the GPU and `"double"` on the CPU, whose kernel is double precision only and raises for `"single"`. Where that sum runs is the package-wide switch `gpu.set_gpu` (see "Running on a GPU"). This is the one sum in the library that a consumer card computes *more slowly* in double precision than the processor does, so on such a card leave it unset

The returned object exposes `qs`, `Wq`, `chi0`, `bare`, `epsmin` (the smallest dielectric eigenvalue found over the mesh), `.at(q)` for the value at a mesh q-point and `.get_dict()` for the inverse Fourier transform back to a real-space interaction. Raises if an eigenvalue of $\varepsilon(q)$ reaches zero, which is a charge or spin instability of the mean field at that wavevector.

## h.get_polarizability()
Return `(qs,chi0)`, the static polarizability of this Hamiltonian on its k-mesh, in the spin-orbital basis; `chi0` has shape `(nq,norb,norb)`. Takes `nk`, `exclude`, `chi_prec` and the precomputed-eigenstate arguments of `get_screened_interaction`.

## h.get_fermi_surface()
Compute the spectral weight on a 2D k-mesh at a single energy.

Optional arguments:

- e=0.0: energy of the cut

- nk=50: k-points per direction

- delta: broadening

- operator=None: project/weight by an operator (e.g. `"sz"`, `"valley"`, `"unfold"`)

- primal_mesh=False: draw the mesh in the Brillouin zone of the primitive cell of a supercell built with `store_primal=True`, the mesh an unfolded Fermi surface is drawn on

Returns kx, ky and the Fermi-surface weight

## h.get_multi_fermi_surface()
Compute the Fermi surface at many energies, writing one file per energy to a `MULTIFERMISURFACE/` folder.

Optional arguments:

- energies=[0.0]: energies to compute

- nk, delta, operator, primal_mesh: as in `get_fermi_surface`

## h.get_surface_kdos()
Compute the surface and bulk spectral function of a semi-infinite system, from the surface Green's function (renormalization/decimation technique).

Optional arguments:

- kpath: k-point path (auto-generated if not given)

- energies, delta: frequency range, broadening

Returns k, energy, surface spectral weight and bulk spectral weight; also writes `KDOS.OUT`

## h.get_qpi()
Compute the quasiparticle-interference map (2D systems only). Returns `(q,energies,qpi)`, with `qpi` one row per energy and one column per wavevector, and also writes the maps to a folder (default `MULTIQPI/`) and the DOS to `DOS.OUT`.

Optional arguments:

- energies, nk, delta: as above

- mode="response": `"pm"` ("poor man's", autoconvolves the actual k-resolved spectral weight, the physical QPI of a real scatterer) or `"response"` (cheaper Lindhard-like joint-DOS convolution of the clean bands)

- nunfold=1: unfold the QPI of a supercell back onto the primitive Brillouin zone, given as the linear size of the supercell as in `g.get_supercell()` (`np.sqrt(3)` for the $\sqrt3\times\sqrt3$ one)

- write=True: `False` returns the arrays without writing any file (see "Files written to the working directory")

## h.get_qpi_impurity()
Compute quasiparticle interference by placing real-space impurities in a supercell, computing the real-space LDOS by partial diagonalization, and Fourier transforming it directly (2D systems only). Returns `(r,ldos_r,q,qpi_q)`.

Optional arguments:

- nsuper=10: supercell size (scalar or `(n1,n2)`)
- impurities=[]: list of dicts, each `{"position"|"index": ..., "onsite": v}` or `{"position"|"index": ..., "vacancy": True}`
- energies=0.0, delta, nk: as above
- num_waves=20: starting number of eigenstates computed nearest the requested energies. It grows automatically until the window reaches `margin` (default 5.0) times `delta` past every requested energy, without stopping inside a degenerate manifold, so too small a value costs time, not correctness
- write=True, output_folder="QPI_IMPURITY": also write the MULTIQPI-style disk output

## h.get_spin_splitting_density()
Compute the energy-resolved spin splitting of a collinear magnet as a smooth weighted density: every band pair contributes its squared splitting, broadened by a Lorentzian of unit area at the mean energy of the pair, so that the integral over energy is the squared splitting summed over bands and averaged over the Brillouin zone. Returns `(energies,values)`.

Optional arguments:

- nk=20: number of k-points per direction
- energies: energies at which the density is evaluated (default 400 points spanning -3 to 3)
- delta=1e-2: broadening
- tol=1e-7: largest spin off-diagonal element of the Bloch Hamiltonian
  tolerated, the same guard `get_spin_splitting_vs_energy` applies; above
  it this raises rather than answering

## h.get_average_spin_splitting()
Return a single number: the spin splitting of a typical band, averaged over
bands as well as over the Brillouin zone. Being an average rather than a
sum it is intensive, so the same crystal described in a larger cell gives
the same answer.

Optional arguments:

- nk=20: number of k-points per direction
- tol=1e-7: collinearity tolerance, as above

Use `h.get_spin_splitting_vs_energy()` instead when what is wanted is the
largest splitting anywhere in the zone rather than the typical one.

## h.get_spin_splitting_vs_energy()
Compute the energy-resolved **maximum** spin splitting over the Brillouin zone, so that the largest value returned bounds the spin splitting anywhere in the zone. Bands are paired by index within each spin channel and binned at the mean energy of the pair. Returns `(energies,values)`, the same convention as `get_spin_splitting_density`, so the two are directly comparable.

Optional arguments:

- nk=100: linear mesh density (nk^d points in d dimensions)
- energies=None: explicit bin centers; otherwise `nbins` points spanning `emin` to `emax`
- nbins=400, emin=None, emax=None: used when `energies` is not given; the default range is that of the mean energies actually found, so no state falls outside the window
- tol=1e-7: largest spin off-diagonal element of the Bloch Hamiltonian tolerated. Above it this raises rather than answering, since the spin-resolved splitting is not meaningful with spin-orbit coupling or non-collinear order

Empty bins are `0.0` (not `NaN`), and states outside an explicitly requested window are dropped rather than clamped onto the end bins. Diagonalization is dense throughout, since a sparse solver would return only the eigenvalues nearest `E=0` and could miss the peak.

## h.get_spin_chern()
Return the spin Chern number $(C_+-C_-)/2$ of a two-dimensional spinful
insulator, with the occupied states split by the sign of $P s_z P$, which
stays quantized when $s_z$ is not conserved (see "Spin Chern number and mirror
Chern number").

Optional arguments:
- operator="sz": the spin component, a name, a matrix or an `Operator`
- nk=40: number of k-points per direction of the mesh
- write=True: `False` keeps the Chern number and Berry curvature of each spin
  sector out of `CHERN.OUT` and `BERRY_CURVATURE.OUT`

It raises `ValueError` if the projected spin closes its gap or the system is
a metal.

## h.get_mirror_chern()
Return the mirror Chern number $(C_{+i}-C_{-i})/2$ of a two-dimensional
insulator with the mirror $z\to -z$, found and verified from the geometry and
the Hamiltonian; it raises `ValueError` when there is no such mirror.

Optional arguments:
- nk=40: number of k-points per direction of the mesh

## h.get_winding_number()
Return the winding number of a one-dimensional Hamiltonian with a chiral
symmetry, the number of zero modes at each end of an open chain (see "Winding
number of a chiral chain").

Optional arguments:
- chiral=None: the chiral operator, a name, a matrix or an `Operator`; by
  default the sublattice operator of a normal Hamiltonian and
  $\sigma_y\tau_y$ on every site of a Nambu one. It raises `ValueError` if
  the operator does not anticommute with the Hamiltonian
- nk=200: number of k-points around the Brillouin zone

## h.get_quadrupole_moment()
Return the quadrupole moment $q_{xy}$ of a two-dimensional insulator with the
mirrors $M_x$ and $M_y$, $0$ or $1/2$, from the Wannier-sector polarizations of
the nested Wilson loops (see "Quadrupole moment and the nested Wilson loop").

Optional arguments:
- nk=40: number of k-points per direction of the mesh
- nocc=None: number of bands, counted from the lowest, taken as occupied; by
  default the ones below zero energy
- tol=1e-4: smallest distance of the Wannier bands from $0$ and $1/2$ that is
  accepted; below it this raises `ValueError`, since the Wannier gap closes

## h.get_wannier_sector_polarization()
Return the polarization along the other direction of one sector of the
Wannier bands, from the nested Wilson loop, modulo 1 in $[-1/4,3/4)$ so that
neither $0$ nor $1/2$ sits at an edge.

Optional arguments:
- loop=0: direction of the first Wilson loop; the polarization is along the
  other one
- sector="+": the Wannier bands with $0<\nu<1/2$, or `"-"` for those with
  $-1/2<\nu<0$; any other name raises `ValueError` listing the two
- gauge="lattice": every orbital at the origin of its cell, or `"atomic"`,
  every orbital at its position
- nk=40, nocc=None, tol=1e-4: as for `h.get_quadrupole_moment()`

## h.get_chern()
Return Chern number of the Hamiltonian.

Optional arguments:
- nk: number of kpoints, 10 for the default `integration="grid"` and 20 for
  `integration="qtci"`; for `integration="wannier"`, the k-points along each
  Wilson loop, 30 by default
- integration="grid": how the Brillouin-zone integral is evaluated. "grid"
  (default) sums the Berry curvature over a uniform nk x nk mesh; "qtci"
  integrates it by quantics tensor cross interpolation plus Gauss-Kronrod
  quadrature, sampling adaptively instead of uniformly; accurate for a
  smooth curvature (a large gap) and unreliable for a sharply peaked one,
  where the mesh sum stays exactly quantized. See
  "Tensor-cross-interpolation (qtci) integration". "wannier" counts the
  winding of the hybrid Wannier centers instead, see "Winding of the hybrid
  Wannier centers"; any other name raises `ValueError` listing the three
- nt=100: for `integration="wannier"`, the number of momenta along the
  second reciprocal direction; a small gap needs more of them
- nocc=None: for `integration="wannier"`, the number of bands counted from the
  lowest whose centers are followed; by default the ones below zero energy
- operator=None: a name, a matrix or an `Operator`, as for
  `h.get_berry_curvature()`. The operator-projected invariants
  (`topology.spin_chern`, `topology.operator_berry`) work on sparse
  Hamiltonians too, which is what a moire or supercell model is
- write=True: also write the Chern number to `CHERN.OUT` and, for
  `integration="grid"`, the Berry curvature on the mesh to
  `BERRY_CURVATURE.OUT`; `False` only returns the number

## h.get_berry_curvature()
Return the Berry curvature of the occupied bands as a map over the
Brillouin zone, `(kx,ky,berry)`. Three flat arrays, so it goes straight
into a `plt.scatter(kx,ky,c=berry)` or, after reshaping to `(nk,nk)`, into
a `contourf`. This is the same curvature that `h.get_chern()` integrates.

Optional arguments:

- nk=100: linear k-point density of the map (the map has `nk*nk` points)

- reciprocal=True: return `kx,ky` in Cartesian reciprocal coordinates,
  which is what you want to plot a hexagonal Brillouin zone undistorted.
  Pass `reciprocal=False` for fractional coordinates instead, in which the
  curvature integrates to the Chern number directly: with `nsuper=1` the
  map covers `[-1,1)` along both fractional directions, i.e. four
  Brillouin zones, so `np.sum(berry)*(2/nk)**2/(2*np.pi)` comes out at
  four times `h.get_chern()`

- mode="Wilson": how the curvature is evaluated. `"Wilson"` uses the
  Fukui-Hatsugai-Suzuki plaquette construction; `"Green"` uses the
  Green's-function Kubo formula and is selected automatically when
  `operator` is given

- operator=None: restrict the curvature to a subspace, e.g. `"valley"` for
  a valley-resolved curvature (see "Berry curvature operator"). A name, a
  matrix or an `Operator` are all accepted, the same spellings
  `h.get_bands(operator=...)` takes; the same holds for `h.get_chern()`,
  `topology.chern_density` and `topology.chern_qtci`

- nsuper=1: extend the map over this many Brillouin zones

- kpath: compute along a k-path instead of over a 2D grid

- delta=0.001: broadening used by the Green's-function mode

## h.get_quantum_geometric_tensor()
Return the (multiband/multiorbital) quantum geometric tensor at a single
k-point, see "Quantum geometric tensor (multiorbital/multiband)".

Optional arguments:
- k=[0.,0.,0.]: k-point
- occ_idxs=None: band indices of the chosen subspace (default: the bands
  with E<0, the same Fermi-level convention `h.get_chern()` uses, so this
  tracks `h.shift_fermi(...)`)
- non_abelian=False: if True, return the full tensor in the orbital basis,
  $\sum_{m,n\in S}|u_m\rangle Q^{mn}\langle u_n|$, an $n\times n$ matrix per pair of
  directions that does not depend on the basis chosen inside a degenerate
  subspace, instead of its trace over the subspace
- degeneracy_tol=1e-8: energy tolerance used to detect a degeneracy
  between the chosen subspace and its complement (raises `ValueError`)
- gauge="atomic": `"atomic"` places every orbital at its position in the
  geometry, the physical quantum geometry; `"lattice"` drops the positions
  from the Bloch phase, as if every orbital sat at the origin of its cell
- coordinates="reduced": `"reduced"` takes the derivatives along the reduced
  momenta of the reciprocal lattice vectors; `"cartesian"` along the
  Cartesian momentum, a $3\times3$ tensor in $x$, $y$ and $z$

## h.get_quantum_metric()
Same arguments as `h.get_quantum_geometric_tensor()`, but returns only the
quantum metric (symmetric part of the tensor).

## h.get_dvector_non_unitarity()
Non-unitarity vector $\vec q = i(\vec d \times \vec d^*)$ of the spin-triplet
d-vector of a BdG (Nambu) Hamiltonian, resolved per site (see "Spin-triplet
d-vector and non-unitary superconductivity"). It is real, vanishes for a
unitary state, and otherwise is the spin moment of the Cooper pairs,
parallel to the magnetization in a ferromagnetic spin-triplet
superconductor, and along $+z$ for a pure $\Delta_{\uparrow\uparrow}$
pairing.

Optional arguments

- nk = 10: k-points per periodic direction

Returns an array of shape (number of sites, 3).

## h.write_non_unitarity()
Write $\vec q$ as a real-space map in `NON_UNITARITY_MAP.OUT`, with columns
$(x,y,z,q_x,q_y,q_z)$.

Optional arguments

- nrep = 2: number of replicas written per direction
- nk = 10: k-points per periodic direction

## h.get_average_dvector()
k-averaged squared components $(|d_x|^2,|d_y|^2,|d_z|^2)$ of the
spin-triplet d-vector of a BdG (Nambu) Hamiltonian.

Optional arguments

- nk = 10: k-points per periodic direction
- spatial_sum = True: average over sites, returning a single 3-component vector
- non_unitarity = False: average the squared components of $\vec q$ instead of those of $\vec d$. This is a magnitude only and carries no sign, so use it to ask whether the state is non-unitary rather than in which direction; use `h.get_dvector_non_unitarity()` for the signed vector.

## h.get_superfluid_weight()
Superfluid weight tensor $D_s^{ab}$ of a BdG (Nambu) Hamiltonian of
dimensionality 1, 2 or 3 (see "Superfluid weight and BKT temperature").

Optional arguments:

- nk=20: k-points per periodic direction
- T=0.0: temperature (see the caveat above for gapless normal states at T=0)
- mode="kubo": `"kubo"` for the analytic multiband formula, `"finite_difference"` for the assumption-free numerical derivative of the grand potential
- gauge="atomic": `"atomic"` twists with the full bond vector (the physical Peierls substitution), `"lattice"` with the lattice vector alone (the cell-gauge convention)
- decompose=False: if True, return a dict with `"total"`, `"conventional"`, `"geometric"`, `"delta"` and `"gauge"` instead of a bare tensor

Returns a Cartesian `(dim,dim)` array, or a dictionary if `decompose=True`.
`decompose=True` raises `ValueError` when the decomposition's assumptions
do not hold, and with `mode="finite_difference"`, which has no such split.

## h.get_bkt_temperature()
Berezinskii-Kosterlitz-Thouless temperature of a 2d BdG Hamiltonian, from
the self-consistent Nelson-Kosterlitz criterion
$T_{\rm BKT} = (\pi/8)D_s(T_{\rm BKT})$ at frozen $|\Delta|$. Arguments
`nk=20`, `tmax=None`, `tol=1e-6`, `maxite=60`, `maxexpand=20`, plus `gauge`.
`tmax` is the upper end of the bisection, $(\pi/8)D_s(0)$ when not given;
where the stiffness is still above the line there, the bracket is doubled up
to `maxexpand` times, and a stiffness that never crosses the line raises
`ValueError`. Returns a float.

## h.get_nonlinear_drude_conductivity()
Compute the l-th order nonlinear Drude conductivity `sigma^{x^l1 y^l2 ; b}` of a collinear magnet, the quantity whose lowest nonvanishing order measures the X-wave index of an altermagnet (d:1, f:2, g:3, i:5; a p-wave magnet has none at any order, and `l = 0` is zero for any Hamiltonian). Requires spin to be a good quantum number.

Optional arguments

- field = "x": one character per power of the electric field, e.g. "yyyyy" for the fifth-order response to E_y
- current = "x": Cartesian direction of the measured current
- channel = "spin": "spin", "charge", "up" or "dn"
- nk = 100: k-points per periodic direction
- T = 0.01: temperature of the Fermi occupations
- mu = 0.0: chemical potential
- tau = 1.0: relaxation time; the l-th order response scales as tau^l
- omega = 0.0: frequency, entering as 1/(i omega + 1/tau)^l
- degeneracy_tol = 1e-8: multiorbital cells only, the relative band spacing below which bands are grouped into a degenerate multiplet

Returns a complex scalar

## h.get_nonlinear_drude_components()
Every component of the l-th order nonlinear Drude conductivity, as a dict `{"x^l1 y^l2;b": value}`. Shares one k-mesh across the components, so it is much cheaper than the equivalent individual calls.

## h.get_nonlinear_drude_orders()
X-wave selection-rule sweep: for each order l = 0..lmax, the largest `|sigma^{x^l1 y^l2 ; b}|` over that order's components. The lowest l with a nonzero entry is the wave index readout.

Optional arguments

- lmax = 6: highest order to sweep
- plus every optional argument of h.get_nonlinear_drude_conductivity()

Returns a list of floats

## h.get_optical_conductivity()
Frequency-dependent conductivity tensor $\sigma_{ab}(\omega)$ in the
Kubo-Greenwood formalism (see "Optical conductivity").

Optional arguments:

- energies=None: frequencies at which to evaluate $\sigma$
- nk=20: k-points per periodic direction
- T=None: temperature (defaults to `delta`)
- delta=0.1: Lorentzian broadening $\eta$
- intraband=True, interband=True: switch the two channels independently
- component=None: e.g. `"xy"` to return that component alone
- degeneracy_tol=1e-6: relative tolerance for treating a pair as degenerate

Returns `(energies,sigma)` with `sigma` of shape `(nw,3,3)` and complex, or
`(nw,)` if `component` is given. Units of $e^2/\hbar$ (so $e^2/h$ is
$1/(2\pi)$ of it). Raises `NotImplementedError` for 3d and Nambu
Hamiltonians.

## h.get_drude_weight()
Drude (intraband) weight tensor. Arguments `nk=20`, `T=0.05`,
`degeneracy_tol=1e-6`. Returns a real `(3,3)` array.

## h.get_sum_rule_weight()
Diamagnetic weight tensor $W$ entering the optical f-sum rule
$\int \mathrm{Re}\,\sigma_{aa}(\omega)\,d\omega = \pi W_{aa}$. Arguments
`nk=20`, `T=0.05`. Returns a real symmetric `(3,3)` array.

## h.get_entanglement_entropy()
Entanglement entropy of a real-space region, from the eigenvalues of the
region-restricted one-particle correlation matrix (see "Entanglement
entropy and entanglement spectrum").

Optional arguments:

- region=None: the region, as a list of site indices, a boolean mask, a callable on positions (the `sculpt` convention) or a float fraction of the cells; `None` takes half the system
- nsuper=10: unit cells stacked into the ring that is cut (periodic Hamiltonians)
- direction=None: lattice direction normal to the cut, defaults to the last periodic one
- kpar=None: momentum parallel to the cut; `None` on a 2d Hamiltonian averages over an `nk` mesh
- nk=20: k-points used for that average
- fermi=0.0: occupied states are those with `E<fermi` (must stay 0 for BdG)

Returns a float. A level exactly at the Fermi energy raises rather than
returning the entropy of an arbitrary determinant.

## h.get_entanglement_spectrum()
Single-particle entanglement Hamiltonian eigenvalues
$\xi_n = \ln[(1-\zeta_n)/\zeta_n]$ of a real-space region. Same arguments as
`h.get_entanglement_entropy()` (with `nk=41` by default).

Returns a sorted array of $\xi_n$ for a 0d/1d Hamiltonian or a single
`kpar`; for a 2d Hamiltonian with `kpar=None` it returns `(ks,xis)` with
`xis` of shape `(nk,nA)`, the Li-Haldane entanglement spectrum across the
BZ.

## h.get_wannier_hamiltonian()
Wannierize a range of bands and return the resulting real-space
Hamiltonian.

Arguments:

- bands = [a,b]: first and last band to Wannierize (0-indexed, both ends inclusive)

Optional arguments:

- nk=12: k-points per periodic direction for the wannierization mesh
- cutoff=1e-6: real-space hoppings smaller than this are dropped (the intracell block is always kept). Set it to `0.0` to keep the reproduction exact to machine precision
- symmetries=None: `"auto"` to auto-detect and enforce the point group, or an explicit list of `symmetrytk.pointgroup.SymmetryOperation`
- num_wann=None: how many Wannier functions to extract. `None` means the whole selected range, Wannierized as a fixed subspace and reproduced exactly. A smaller value switches on Souza-Marzari-Vanderbilt disentanglement, after which only the frozen window is reproduced (see "Disentanglement"). Outside `1..len(bands)` raises `ValueError`
- dis_win_min, dis_win_max=None: outer energy window, i.e. which bands are offered to the extraction at each k-point. Every eigenvalue of the result lies inside it
- dis_froz_min, dis_froz_max=None: frozen inner window, whose states are reproduced exactly at every mesh k-point. `dis_froz_max` alone is enough; `dis_froz_min` alone raises `ValueError`
- dis_num_iter=200: maximum iterations of the $\Omega_I$ minimization; not converging within them is a warning, not an error

Any `dis_*` argument given without a smaller `num_wann` raises
`ValueError`. Disentanglement together with a Nambu/BdG
Hamiltonian, with `symmetries=`, or with `auto_split_clusters=True` raises
`NotImplementedError`.

Returns a new, smaller Hamiltonian; `.wannier_centres`, `.wannier_spreads`
and `.wannier_spread_total` hold the Wannier-function geometry, and
`.wannier_num_wann`/`.wannier_disentanglement_window` record what was
extracted and through which window (the latter `None` when not
disentangling)

## h.get_szsz_mean_field_hamiltonian()
Self-consistent Hartree-Fock mean field for a $J_z\sum S^z_iS^z_j$
spin-spin exchange interaction (see "Spin-spin exchange interactions").
$J>0$ is antiferromagnetic, $J<0$ ferromagnetic.

Optional arguments:

- J1, J2, J3 = 0.: first/second/third-neighbor $J_z$ couplings
- Jr=None: general distance-dependent coupling function, as `Vr` for `get_mean_field_hamiltonian`; `Jr(0)` is an onsite $J_r(0)S^z_iS^z_i/2$, so a bond coupling returns zero at zero distance
- filling, mf, nk, maxerror, mix, constrains: as in `get_mean_field_hamiltonian`
- return_total_energy=False: also return the total energy

Also works on BdG (Nambu) Hamiltonians, decoupling both the normal and
anomalous (pairing) channels.

Returns the converged Hamiltonian (or `None` if the SCF did not converge)

## h.get_sxsx_mean_field_hamiltonian() / h.get_sysy_mean_field_hamiltonian()
Same as `get_szsz_mean_field_hamiltonian()`, for a $S^x_iS^x_j$ /
$S^y_iS^y_j$ interaction instead, implemented by rotating the problem so
that x (or y) becomes the computational z axis, solving there, and
rotating the converged Hamiltonian back. Also works on BdG Hamiltonians.

## h.get_exchange_mean_field_hamiltonian()
Self-consistent anisotropic exchange mean field, combining
$J_x S^x_iS^x_j + J_yS^y_iS^y_j + J_zS^z_iS^z_j$ in a single SCF loop.

Optional arguments:

- Jx1, Jx2, Jx3, Jy1, Jy2, Jy3, Jz1, Jz2, Jz3 = 0.: first/second/third-neighbor couplings for each axis
- Jxr, Jyr, Jzr=None: general distance-dependent couplings, one per axis
- mf, filling, nk, maxerror, mix, constrains: as above (only `integration="ed"` and the plain-mixing solver are supported)

Also works on BdG Hamiltonians, with the same full normal-plus-anomalous
decoupling as `get_combined_mean_field_hamiltonian`'s density-density
channels. Exchange can itself induce superconducting pairing (e.g. an
antiferromagnetic isotropic $J$ alone, seeded with a random guess, can
spontaneously decouple into a purely superconducting state).

Returns the converged Hamiltonian (or `None` if the SCF did not converge)

## h.get_combined_mean_field_hamiltonian()
Self-consistent mean field combining density-density interactions
(onsite $U$, $V_1$/$V_2$/$V_3$/$V_r$ neighbor-shell) with spin-spin
exchange in a single SCF loop.

Optional arguments:

- U, V1, V2, V3, Vr: as in `get_mean_field_hamiltonian`
- J1, J2, J3 = 0.: isotropic Heisenberg exchange for the first/second/third-neighbor shells (same shell convention as V1/V2/V3)
- Jr=None: general distance-dependent isotropic exchange function `Jr(r1,r2)`, as `Vr`; a site with itself is one of the pairs, so `Jr(0)` is an onsite $J_r(0)\,\vec S_i\cdot\vec S_i/2$, a Hubbard $U=-3J_r(0)/4$ at the mean-field level, and a bond coupling returns zero at zero distance
- rcut=None: range of `Vr` and `Jr`, every pair of sites up to that distance and none beyond it, the pair of a site with itself included (`Vr(0)` is an onsite $U=V_r(0)$); `None` means 5 in a periodic system and every pair in a finite one
- J1x, J1y, J1z = 0.: optional anisotropic correction added to J1 on the first-neighbor shell only (e.g. the effective first-neighbor Jz coupling is J1+J1z); second/third neighbors stay purely isotropic

- mf, filling, nk, maxerror, mix, constrains: as above (only the plain-mixing solver is supported)
- integration="ed": compute the density matrix at each SCF iteration by
  exact diagonalization. `"kpm"` uses a per-k Chebyshev-moment (Kernel
  Polynomial Method) expansion instead, never diagonalizing the Bloch
  Hamiltonian $H(k)$, for systems too large to diagonalize or to hold as
  a dense matrix (a big 0D flake, say). A Hamiltonian built with
  `is_sparse=True` keeps every matrix of the loop sparse, with memory
  linear in the number of sites, and is the only way to use `"kpm"` with a
  Nambu Hamiltonian. With `"kpm"`, `scale=None` sets the KPM energy
  rescaling (estimated automatically if not given), `npol` the number of
  Chebyshev moments, `ne` the number of energies sampled in the occupied
  window, `cores` the number of parallel workers across k-points and
  `kpm_radius` the number of hops between sites the recursion of every
  orbital is truncated to (`None`, the default, for the whole system), which
  makes the time linear in the number of sites and is controlled in a gapped
  state or at a finite temperature; all five are unused for `"ed"`. A `scale` given by hand has to cover the
  spectrum of the Hamiltonian after the Fermi shift, not only the bare band
  structure, and one that does not raises a `ValueError`. Also reachable
  through `h.get_mean_field_hamiltonian(integration="kpm",...)`.
- `use_jax=True, solver=...`: solve the same SCF fixed point $x=f(x)$ ($x$
  the mean-field parameters, $f$ one SCF iteration) with a nonlinear solver
  instead of plain mixing. `"error_gradient"` is the most robust of these on
  a generic Hamiltonian, `"newton"` the default; as local methods they can
  still stall short of `maxerror`, in which case the call returns `None`
  like any other non-converged SCF, so always check for that. Where
  `"newton"`, `"newton_krylov"` or a stalled `"fsolve"` sit at a stationary
  point of the residual, they leave it with bursts of `kick_steps=60`
  linear-mixing steps, a length tuned on one system, so a case that still
  stalls can be tried with another. Restricted
  to a normal-state (non-BdG) Hamiltonian, dense exact diagonalization only
  (no `integration="kpm"`), and no `constrains`:
```python
from pyqula import geometry
h = geometry.chain().get_hamiltonian()
hmf = h.get_combined_mean_field_hamiltonian(U=4.0,J1=-0.5,filling=0.5,
        use_jax=True,solver="newton") # JAX-derivative-based SCF solver
```

On a BdG Hamiltonian every channel keeps the full normal+anomalous
(pairing) treatment: $U$/$V_1$/$V_2$/$V_3$/$V_r$ as in
`get_mean_field_hamiltonian`, and the exchange ($J$) channels identically,
so exchange alone can induce superconducting pairing rather than only
magnetism (see "Spin-spin exchange interactions" above).

Returns the converged Hamiltonian (or `None` if the SCF did not converge)

`filling` also accepts a per-site array (length `len(h.geometry.r)`, same
0-to-1 fraction of the two states per site as the scalar case) instead of
a single lattice-averaged value, enforcing $\langle n_i\rangle=$
`filling[i]` at every site independently through a per-site Lagrange
multiplier co-converged with the mean field in the same SCF loop. The
converged per-site potentials are left on the `fermi` attribute of the
returned Hamiltonian, and a `None` return means that either the mean field
or the per-site constraint did not converge to within `maxerror`. Only
supported for a normal-state (non-BdG), `integration="ed"` Hamiltonian with
`mu=None` (the default). This is the mechanism `SpinonHamiltonian` (see
"Abrikosov-pseudofermion (spinon) mean field for Heisenberg models") builds
on to enforce exactly one auxiliary fermion per site.

## SpinonHamiltonian(g)
Abrikosov-pseudofermion (RVB) mean-field Hamiltonian for a spin-$\tfrac12$
Heisenberg model on geometry `g`. See "Abrikosov-pseudofermion (spinon)
mean field for Heisenberg models" above. `from pyqula.spinon import
SpinonHamiltonian`; built with zero bare hopping, couplings supplied
through `get_mean_field_hamiltonian`'s usual `J1`/`J2`/`J3`/`Jr`/`J1x`/
`J1y`/`J1z` kwargs.

- `h.get_mean_field_hamiltonian(J1=...,nk=...,...)`: same SCF kwargs as
  `get_combined_mean_field_hamiltonian` above, except `filling` cannot be
  passed (always exactly one fermion/site, enforced site-by-site). Returns
  the converged Hamiltonian (or `None` if the SCF did not converge), with
  two extra diagnostic attributes:
  - `h2.local_occupation`: converged $\langle n_i\rangle$ per site
    (electron-count convention, 0 to 2, target is exactly 1.0)
  - `h2.constraint_lambda`: converged per-site Lagrange multiplier (local
    chemical potential)

## KondoLatticeHamiltonian(hc)
Abrikosov-pseudofermion (Read-Newns) mean-field Hamiltonian for the Kondo
lattice / periodic Anderson model built from a conduction-electron
Hamiltonian `hc`. See "Abrikosov-pseudofermion (Read-Newns) mean field
for the Kondo lattice" above. `from pyqula.kondolattice import
KondoLatticeHamiltonian`; fuses a second, zero-bare-hopping f-sublattice
onto `hc`'s geometry, with the Kondo coupling supplied through
`get_mean_field_hamiltonian`'s `J` kwarg.

- `h.get_mean_field_hamiltonian(J=...,filling=...,mf=(V,lam),nk=...,...)`:
  self-consistently solves for the hybridization and the local
  constraint's Lagrange multiplier. Returns the converged Hamiltonian (or
  `None` if the SCF did not converge), with three extra diagnostic
  attributes:
  - `h2.local_occupation`: converged $\langle n_f\rangle$ per localized
    site (target is exactly 1.0)
  - `h2.hybridization`: converged $V_j$ per localized site
  - `h2.constraint_lambda`: converged per-site Lagrange multiplier

## GrapheneGeometry(g)
`Geometry` subclass wrapping a graphene multilayer geometry `g` (bilayer,
twisted bilayer, twisted trilayer, ...), adding a `.relax()` method, see
"Twisted bilayer graphene structural relaxation" above. `from
pyqula.graphenetk.geometry import GrapheneGeometry`; `g` must have
`has_sublattice=True` (raises `ValueError` otherwise).

- `g.relax(nrep=1,maxiter=500,verbose=False,layer_pairs=None,gsfe_coeffs=...,elastic_coeffs=...)`:
  minimizes the GSFE (interlayer) + elastic (intralayer) energy over an
  in-plane displacement field and returns a new, relaxed
  `GrapheneGeometry`. `layer_pairs` selects which (lower,upper) pairs of
  layers (ordered by z) get an interlayer GSFE term, defaulting to all
  adjacent pairs; `gsfe_coeffs`/`elastic_coeffs` override the built-in
  graphene constants.

## GrapheneHamiltonian(geometry)
Hamiltonian built from a graphene multilayer `geometry` (typically a
`GrapheneGeometry`, relaxed or not), see "Twisted bilayer graphene
structural relaxation" above. `from pyqula.graphenetk.hamiltonian import
GrapheneHamiltonian`; defaults to the distance-decaying hoppings of
`specialhopping.twisted_matrix` (`ti=0.12,lambi=8.0,lamb=12.0,dl=3.0`,
matching `specialhamiltonian.twisted_bilayer_graphene`'s defaults) rather
than the generic first-neighbor default, so relaxed (in-plane displaced)
positions feed into the electronic structure automatically. Pass
`mgenerator=...` to use a different hopping generator instead.

## h.get_central_heterostructure()
Build a two-terminal `Heterostructure` using `h` (a finite, 0d Hamiltonian) as the central scattering region, contacted by two semi-infinite 1D chain leads attached at sites `i`/`j` (see "Transport through an arbitrary finite region" above).

Optional arguments:

- i=0, j=None: 0-indexed sites `h` is contacted at; `j` defaults to the last site

- left=None, right=None: lead Hamiltonians; default to a plain spinless `geometry.chain()`. Give one of them (or `h` itself) nonzero pairing (`add_swave`) for a normal-superconductor junction. At most one of `{h, left, right}` may carry pairing

Returns a `Heterostructure`, so `didv`, `get_dos`, `get_kappa`, etc. all apply unmodified; `landauer` is the exception, raising `NotImplementedError` once any of the three carries pairing (use `didv` there). Only 0d central regions are supported so far (`h.dimensionality>0` raises `NotImplementedError`).
