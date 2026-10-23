# Hamiltonian functions and methods

The methods `h.add_...()` change `h` in place. The methods `h.get_...()`
return a result and leave `h` unchanged. Energies are in units of the
first-neighbor hopping $t$, and the Fermi energy is at zero energy.

## h.get_bands()
Compute the band structure $\epsilon_n(\mathbf k)$ along a k-path. See
[Band structures and DOS](bands-and-dos.md).

Optional arguments:

- nk=400: number of k-points along the path
- operator: an operator, or a list of operators. For each eigenstate the
  expectation value $\langle \Psi_n(\mathbf k)|A|\Psi_n(\mathbf k)\rangle$ is
  also returned
- kpath: the k-path, as a list of k-points in reduced coordinates or as a
  list of high-symmetry labels (`"G"`, `"M"`, `"K"`, `"X"`, `"Y"`, and in
  three dimensions `"Z"`, `"R"`, `"A"`, `"B"`)
- ewindow: a function of the energy that says which bands to keep, for
  example `ewindow=lambda e: abs(e)<0.5`
- eigmode="complex": non-Hermitian Hamiltonians only, which part of the
  complex eigenvalue is returned, `"complex"`, `"real"` or `"imag"` (see
  [Non-Hermitian Hamiltonians](non-hermitian.md))
- biorthogonal=False: non-Hermitian Hamiltonians only, weight each state by
  $\langle L|A|R\rangle/\langle L|R\rangle$ instead of the right eigenvector
  alone
- write=True: also write `BANDS.OUT` and the k-path files (see
  [Output files](files.md))

Without `kpath`, the path is $\Gamma$-M for a square-like 2D lattice,
$\Gamma$-K-M-K'-$\Gamma$ for a triangular-like one, and
$\Gamma$-X-M-$\Gamma$-R for a 3D one.

Returns the index of the k-point along the path and the energies, plus one
array per operator if `operator` is given

## h.get_kdos_bands()
Compute the momentum-resolved spectral function along a k-path,

$$A(\mathbf k,\omega) = -\frac{1}{\pi}\,\mathrm{Im}\,\mathrm{Tr}\left[A\,G(\mathbf k,\omega+i\delta)\right],$$

with $A$ the identity or a projection operator. This is what ARPES
measures. See [Spectral functions](spectral-functions.md) and
[Band unfolding](unfolding.md).

Optional arguments:

- kpath: the k-path (generated if not given)
- operator=None: operator that weights the spectral function, for example
  `"unfold"` to unfold the bands of a supercell
- energies, delta, nk: energies, broadening $\delta$ and number of
  k-points
- mode="ED": `"ED"` (diagonalization), `"green"` (Green's function) or
  `"KPM"` (Chebyshev expansion, for large unit cells)
- frand=None: generator of the random vectors of the KPM trace, only with
  `mode="KPM"`
- biorthogonal=False: non-Hermitian Hamiltonians only, use the
  Green's-function spectral function
- write=True: also write `KDOS_BANDS.OUT`

Returns the position along the k-path, the energy and the spectral weight

## h.get_dos()
Compute the density of states

$$\rho(\omega) = \frac{1}{N_k}\sum_{n,\mathbf k}\delta(\omega-\epsilon_n(\mathbf k)),$$

with the delta function broadened by a Lorentzian of width $\delta$. Its
integral over energy is the number of states per unit cell. See
[Band structures and DOS](bands-and-dos.md).

Optional arguments:

- energies: array of energies
- delta=None: broadening. By default `5/nk`, so 0.05 for `nk=100`. A
  broadening smaller than the level spacing of the k-mesh shows the mesh,
  not the physics
- mode="ED": `"ED"`, `"KPM"`, `"adaptive"`, `"Green"` or `"RG"` (see
  [Band structures and DOS](bands-and-dos.md))
- nk=100: k-points per direction
- operator=None: operator the DOS is projected on, a name, a matrix or an
  `Operator`

For a non-Hermitian Hamiltonian only `mode="ED"` exists (see
[Non-Hermitian Hamiltonians](non-hermitian.md)).

Returns energies and DOS

## h.get_gap()
Return the indirect gap, the smallest energy difference between an empty and
an occupied state anywhere in the Brillouin zone. The two states can be at
different k-points. The band edges are found by minimizing over $\mathbf k$,
not on a fixed mesh, so a gap that closes at a generic k-point is found.

Optional arguments:

- ntries=1: number of minimizations; the smallest result is kept

Returns a number. It is zero for a metal or a Dirac semimetal

## h.get_bandwidth()
Return the bottom and the top of the spectrum, `(emin,emax)`. Note that it
returns the two band edges, not their difference.

## h.get_filling()
Return the filling, the fraction of the states below zero energy,

$$\nu = \frac{1}{N N_k}\sum_{n,\mathbf k}\Theta(-\epsilon_n(\mathbf k)),$$

with $N$ the number of states per unit cell. Half filling is 0.5. Use
`h.set_filling(nu)` to fix a filling, and this method to check it (see
[Building a Hamiltonian](hamiltonian.md)).

For a Nambu Hamiltonian the spectrum is particle-hole symmetric, so half of
it is always below zero. There the filling is computed from the electron
part of the occupied states, so that with zero pairing it is the filling of
the normal state.

Optional arguments:

- nk: k-points per direction

## h.get_total_energy()
Return the total energy of the occupied states,

$$E = \frac{1}{N_k}\sum_{n,\mathbf k}\epsilon_n(\mathbf k)\,\Theta(\epsilon_F-\epsilon_n(\mathbf k)),$$

per unit cell. For a mean-field Hamiltonian this is only the band energy;
the interacting energy, with the double counting removed, is returned by
`h.get_mean_field_hamiltonian(...,return_total_energy=True)`.

Optional arguments:

- nk=10: k-points per direction
- fermi=0.0: the Fermi energy $\epsilon_F$. Not accepted for a Nambu
  Hamiltonian, because a chemical potential shifts the electron and hole
  parts in opposite directions; use `h.shift_fermi(-mu)` instead
- mode="mesh": sum on a k-mesh; `use_kpm=True` uses a Chebyshev expansion
  instead, for large systems

For a Nambu Hamiltonian it returns the energy of the electrons, so with
zero pairing it is the energy of the normal state.

## h.get_density_matrix()
Return the density matrix of the occupied states, as a matrix in the basis
of the Hamiltonian. See [Mean-field Hubbard models](mean-field.md).

Note the convention: $\mathrm{dm}_{ij}=\sum_\mathrm{occ}\psi_i^*\psi_j$, the
transpose of the usual $\rho_{ij}=\sum_\mathrm{occ}\psi_i\psi_j^*$. So an
expectation value is `np.trace(dm.T@A)`, not `np.trace(dm@A)`. The two are
the same for a real operator (the density, $\sigma_x$, $\sigma_z$) and
differ by a sign for a purely imaginary one ($\sigma_y$, the valley
operator, a current). It is simpler to use `h.get_vev(operator)`.

## h.get_ipr()
Return the inverse participation ratio of every eigenstate,

$$\mathrm{IPR}_n = \sum_i |\psi_n(i)|^4 ,$$

as `(energies,ipr)`. A state spread over $N$ sites has
$\mathrm{IPR}\sim 1/N$, and a state on a single site has
$\mathrm{IPR}\sim 1$, so it tells localized from extended states, for
example in Anderson localization or for in-gap bound states. Only for finite
(0D) systems; a periodic Hamiltonian raises `NotImplementedError`, and
there the IPR operator can be used instead (see [Operators](operators.md)).

## h.get_vev() / h.get_single_vev() / h.get_several_vev()
Ground-state expectation values of operators, summed over the occupied
states.

- `h.get_vev(operator=...)` returns one real number per site, the
  expectation value of `operator` on each site, for example `"sz"` for a
  magnetization map. Without `operator` it returns the occupation of each
  site
- `h.get_single_vev(A)` returns one number, $\langle A\rangle$ summed over
  the whole system
- `h.get_several_vev([A,B,...])` returns $\langle A\rangle$,
  $\langle B\rangle$, ... with a single diagonalization

An operator that depends on $\mathbf k$, like `"unfold"`, is applied inside
the sum over the Brillouin zone, so `h.get_vev(operator="unfold")` gives the
unfolded weight on each site:

```python
from pyqula import geometry
g = geometry.honeycomb_lattice() # primitive geometry
gs = g.get_supercell(2,store_primal=True) # supercell, keeping the primitive cell
h = gs.get_hamiltonian(has_spin=False)
h.add_onsite(lambda r: 100.0 if sum((r-gs.r[0])**2)<1e-2 else 0.0) # a vacancy
d = h.get_vev(operator="unfold") # unfolded weight on each site
```

For a Nambu Hamiltonian a normal observable is counted once, so with zero
pairing the result is the one of the normal Hamiltonian. A pairing operator
like `"spair"` gives the anomalous expectation value, the local s-wave
pairing amplitude, per site with
`h.get_vev("spair")` and summed with `h.get_single_vev("spair")`.

Optional arguments:

- nk=30: k-points per direction

## h.get_magnetization()
The magnetization of each site, as an `(nsites,3)` array. There are two
quantities with this name, chosen by `mode`:

- `mode="vev"` (default): the spin expectation value of each site,
  $(\langle \sigma_x\rangle,\langle \sigma_y\rangle,\langle \sigma_z\rangle)$,
  summed over the occupied states. It needs a k-mesh, given by `nk` or the
  one remembered from a self-consistent calculation. In a metal at zero
  temperature it changes in steps of order $1/n_k$, so use a fine mesh to
  compare weakly polarized states
- `mode="field"`: the magnetic term written in the Hamiltonian, the
  coefficients of $\sigma_{x,y,z}$ on each site. After a mean-field
  calculation this is the exchange field, proportional to the moment

Note the sign: `h.add_zeeman()` and `h.add_exchange()` add
$+\vec B\cdot\vec\sigma$, so the electrons lower their energy with the spin
against $\vec B$, and the moment is antiparallel to the field:

```python
from pyqula import geometry
h = geometry.chain().get_hamiltonian()
h.add_exchange([0.,0.,0.5])
h.get_magnetization(nk=40)         # the moment: [0,0,-0.15], against the field
h.get_magnetization(mode="field")  # [0,0,0.5], the field you put in
```

Optional arguments:

- mode="vev": `"vev"` or `"field"`
- other keywords (for example `nk`) are passed to `h.get_vev()` with
  `mode="vev"`


## h.get_topological_invariant()
Return the topological invariant of the occupied bands that fits the
dimension and the symmetry of the Hamiltonian:

- 1D: the $Z_2$ invariant ($\pm1$) of a time-reversal symmetric
  superconductor (class DIII), and otherwise the Berry (Zak) phase
- 2D: the $Z_2$ invariant if the Hamiltonian is time-reversal symmetric
  (also a BdG Hamiltonian), and otherwise the Chern number
- 3D: the strong and weak $Z_2$ indices `(nu0, (nu1, nu2, nu3))`, each 0 or
  1, if it is time-reversal symmetric, and otherwise the Chern vector
  `(C1, C2, C3)`

A 0D Hamiltonian raises `ValueError`. See
[Topological invariants](topological-invariants.md).


## h.add_soc()
Add the Kane-Mele spin-orbit coupling, the same as `h.add_kane_mele()`.

Arguments:

- t: strength of the spin-orbit coupling


## h.add_zeeman()
Add a Zeeman field $\sum_i \vec B_i\cdot\vec\sigma$ (see
[Building a Hamiltonian](hamiltonian.md)).

Arguments:

- zeeman: the field $\vec B$, as a number (meaning $(0,0,B_z)$), a 3-vector,
  one vector per site, or a function of the position


## h.add_rashba()
Add the Rashba spin-orbit coupling between first neighbors,

$$H_R = i\lambda_R\sum_{\langle ij\rangle} c^\dagger_i\,
\hat z\cdot(\mathbf r_{ij}\times\vec\sigma)\, c_j ,$$

with $\mathbf r_{ij}$ the bond vector. It comes from a field perpendicular
to the plane, which breaks the mirror symmetry $z\to-z$.

Arguments:

- c: the Rashba coupling $\lambda_R$, a number or a function of the
  position


## h.add_onsite()

Add an onsite energy $\sum_i \mu_i c^\dagger_i c_i$ (see
[Building a Hamiltonian](hamiltonian.md)).

Arguments:

- fermi: the onsite energy, a number, an array with one value per site, or
  a function of the position


## h.add_sublattice_imbalance()
Add a staggered onsite energy, $+m$ on one sublattice and $-m$ on the other,

$$H_m = m\sum_i \tau_i\, c^\dagger_i c_i ,\qquad \tau_i=\pm1 .$$

In the honeycomb lattice this opens a gap $2m$ at the Dirac points, as in
boron nitride.

Arguments:

- mass: the imbalance $m$, a number or a function of the position

The geometry needs a sublattice with two values. A chain, a square lattice
or a kagome lattice raise `ValueError`. For a bipartite lattice without
sublattice you can use `g = g.get_supercell(2)` and then
`g.get_sublattice()`; otherwise write the potential with `h.add_onsite(f)`.


## h.add_antiferromagnetism()
Add a staggered exchange field: a Neel pattern $\pm m\,\sigma_z$ on two
sublattices, and a 120-degree pattern on three or more.

Arguments:

- mass: the exchange field, a number or a function of the position

The geometry needs a sublattice; without one it raises `ValueError`.

## h.add_haldane()
Add the Haldane term, an imaginary second-neighbor hopping,

$$H_H = i t\sum_{\langle\langle ij\rangle\rangle}\nu_{ij}\, c^\dagger_i c_j ,
\qquad \nu_{ij} = (\mathbf d_1\times\mathbf d_2)_z ,$$

where $\mathbf d_1$ and $\mathbf d_2$ are the two first-neighbor bonds that
go from $i$ to $j$. In the honeycomb lattice $\nu_{ij}=\pm\sqrt3/2$, the
sign given by the direction of the turn. It breaks time-reversal symmetry
and opens a gap of $9t$ at both Dirac points, and the result is a Chern
insulator with $C=\pm1$ (see
[Topological invariants](topological-invariants.md)).

Arguments

- t: the second-neighbor hopping $t$, a number or a function of the
  position

## h.add_modified_haldane() / h.add_antihaldane()
The same hopping as `h.add_haldane()`, but with an extra sign $\tau_i$
between the two sublattices. The two valleys get opposite masses and the
Chern number is zero: it is a valley Hall insulator instead of a Chern
insulator. The two names are the same method.

Arguments

- t: the second-neighbor hopping

## h.add_kane_mele()
Add the Kane-Mele spin-orbit coupling, the Haldane hopping with opposite
sign for the two spins,

$$H_{KM} = i t\sum_{\langle\langle ij\rangle\rangle}\nu_{ij}\, c^\dagger_i \sigma_z c_j .$$

It keeps time-reversal symmetry, and the honeycomb lattice becomes a quantum
spin Hall insulator, with gap $9t$ and $Z_2=1$. A spinless Hamiltonian is
made spinful first.

Arguments

- t: the spin-orbit coupling $t$

## h.add_anti_kane_mele()
The Kane-Mele term with the extra sublattice sign $\tau_i$, in the same way
as `h.add_modified_haldane()` is related to `h.add_haldane()`.

Arguments

- t: the second-neighbor hopping

## h.add_kekule() / h.add_chiral_kekule()
Add a Kekule modulation of the first-neighbor hoppings of the honeycomb
lattice, with a period of three bonds. It couples the two Dirac points,
which are folded to $\Gamma$, and opens a gap. `add_kekule(t)` takes the
amplitude of the modulation. `add_chiral_kekule(t1=...,t2=...)` adds the
chiral version, with two independent amplitudes.

## h.add_valley_exchange()
Add a valley exchange field $\vec v\cdot\vec\tau$, with
$\vec\tau=(\tau_x,\tau_y,\tau_z)$ the valley Pauli matrices. It is the
analog of `h.add_exchange()` for the valley pseudospin (see
[Operators](operators.md)).

Arguments

- v = (vx,vy,vz): the valley field

For a periodic Hamiltonian the in-plane components couple the two valleys,
so the unit cell must contain both: a supercell of the honeycomb lattice
with a size multiple of 3. Otherwise it raises `ValueError`. A finite flake
works always.

## h.add_peierls() / h.add_orbital_magnetic_field()
Add an out-of-plane magnetic field as a Peierls phase on every hopping,
$t_{ij}\to t_{ij}e^{i\phi_{ij}}$ (see
[Building a Hamiltonian](hamiltonian.md)). The two names are the same
method.

Arguments

- mag_field: the field $B$, as the flux through a unit area in units of the
  flux quantum. In the Landau gauge the phase on a bond is
  $\phi_{ij}=2\pi B\,y\,\Delta x$. In a periodic system the flux through the
  unit cell must be an integer number of flux quanta, so build a supercell
  first and choose `mag_field` to match it

Optional arguments

- gauge="Landau": `"Landau"` or `"symmetric"`

A Hamiltonian with pairing raises `NotImplementedError`: a Cooper pair has
charge $2e$, so the pairing term picks up a different phase, and a field in
a superconductor means vortices. Add the field to the normal Hamiltonian
first, and then the pairing.

## h.add_inplane_bfield()
Add an in-plane magnetic field, as a Peierls phase that depends on the
out-of-plane coordinate. It matters for a multilayer, where an in-plane
field gives a flux between the layers.

Optional arguments

- b=0.0: strength of the field
- phi=0.0: in-plane direction of the field, in units of $\pi$

Only for 0D, 1D and 2D systems

## h.add_strain()
Change the hoppings according to a strain field. In graphene a
non-uniform strain acts as a pseudo-magnetic field, with opposite sign in
the two valleys.

Arguments

- sr: function of the position that gives the local strain

Optional arguments

- mode="scalar": `"scalar"` multiplies each hopping by `sr(r)`;
  `"directional"` also uses the direction of the bond, for anisotropic
  strain

## h.add_crystal_field()
Add a crystal field: an onsite energy that depends on the neighbors of each
site, so that sites with fewer or farther neighbors (at an edge, next to a
vacancy, or in different stackings of a bilayer) get a different energy.

Arguments

- v: strength of the crystal field. The average is removed, so `v` sets the
  spread of the onsite energies, not their offset

Optional arguments

- rcut=6.0: cutoff distance of the sum over neighbors

## h.generate_spin_spiral()
Rotate the spin quantization axis from cell to cell, with a wavevector
$\mathbf q$, to describe a spin spiral in the unit cell of the lattice
(generalized Bloch theorem). It is used as a starting point for a spiral
mean-field calculation, and to compute the energy of a spiral as a function
of $\mathbf q$ (see [Exchange interactions](exchange.md)).

Optional arguments

- vector=[0.,0.,1.]: the axis around which the spins rotate
- qspiral=[1.,0.,0.]: the wavevector $\mathbf q$ of the spiral
- fractional=True: `qspiral` in reduced coordinates. With `False` only the
  hoppings between cells are rotated

Needs a spinful Hamiltonian

## h.add_pairing()
Add a superconducting pairing term with a given symmetry. The Hamiltonian
becomes a Bogoliubov-de Gennes (BdG) Hamiltonian if it is not already (see
[Superconductivity](superconductivity.md)).

Optional arguments

- delta=0.0: pairing amplitude $\Delta$, a number or a function of the
  position
- mode="swave": the symmetry of the pairing, for example `"swave"`,
  `"extended_swave"`, `"triplet"`, `"pwave"`, `"chiral_pwave"`, `"dx2y2"`,
  `"dxy"`, `"nodal_dwave"`, `"chiral_dwave"`, `"nodal_fwave"`,
  `"chiral_fwave"`, `"chiral_gwave"`. An unknown name raises `ValueError`
  with the list of all accepted ones. It can also be a function of two
  positions that returns the $2\times2$ pairing matrix $D(\mathbf r_1,\mathbf r_2)$.
  This function must respect the Fermi antisymmetry,
  $D(\mathbf r_1,\mathbf r_2)=\sigma_y D(\mathbf r_2,\mathbf r_1)^T\sigma_y$,
  meaning that the singlet part is even and the triplet part odd under
  exchange of the two positions; otherwise it raises `ValueError`
- d=[0.,0.,1.]: the d-vector of a triplet pairing, a 3-vector or a function
  of the position

A `delta` or `d` that depends on the position must have the periodicity of
the lattice, otherwise it raises `ValueError`. A modulated pairing is built
on a supercell that fits the modulation. In a 0D system any function works.

## h.setup_nambu_spinor()
Write the Hamiltonian in the Nambu (electron-hole) basis, with zero pairing.
This is needed before any superconducting calculation on a normal
Hamiltonian, for example a mean field with a pairing channel. It is the
same as `h.add_swave(0.0)`. A Nambu Hamiltonian is always spinful, so a
spinless Hamiltonian is made spinful first. Changes `h` in place

## h.turn_nambu()
Add the electron-hole degree of freedom without pairing, making a spinless
Hamiltonian spinful first. Normally `h.setup_nambu_spinor()` is the one to
use. Changes `h` in place

## h.turn_spinful()
Add the spin degree of freedom to a spinless Hamiltonian; every matrix is
multiplied by the identity in spin space. Changes `h` in place, and does
nothing if `h` is already spinful

## h.remove_spin()
Keep one spin block and remove the other, which gives a spinless
Hamiltonian.

Optional arguments

- channel="up": the block that is kept, `"up"` or `"dn"`

The part of the Hamiltonian that mixes the two spins is dropped, so this is
only meaningful when $s_z$ is conserved. With spin-orbit coupling or
non-collinear magnetism the result is a different model.

## h.turn_multicell()
Store the hoppings as a dictionary indexed by the lattice vector between
the cells, which allows hoppings beyond the neighboring cells. Some methods
need it, and their error messages say so. Changes `h` in place.
`h.get_no_multicell()` returns a copy in the other form when possible

## h.get_hk_gen()
Return the Bloch Hamiltonian as a function of $\mathbf k$,

$$H(\mathbf k)=\sum_{\mathbf R} t(\mathbf R)\, e^{2\pi i\,\mathbf k\cdot\mathbf R},$$

with $\mathbf k$ in reduced coordinates (in units of the reciprocal lattice
vectors), $t(\mathbf R)$ the hopping matrix between cells separated by
$\mathbf R$, and no positions of the sites inside the cell in the phase. For
example `h.get_hk_gen()([0.3,0.,0.])` is $H(\mathbf k)$ at
$\mathbf k=0.3\,\mathbf b_1$.

## h.get_gf()
Return the Green's function of the infinite system at one energy,

$$G(\omega)=\frac{1}{N_k}\sum_{\mathbf k}\left[\omega+i\delta-H(\mathbf k)\right]^{-1},$$

as a matrix in the basis of the unit cell.

Optional arguments

- energy=0.0: the energy $\omega$
- delta=1e-5: the imaginary part $\delta$
- mode="adaptive": how the integral over $\mathbf k$ is done, `"adaptive"`
  (with error control), `"full"` (a fixed mesh of `nk` points) or
  `"renormalization"`
- gtype="bulk": `"bulk"` or `"surface"` Green's function

$-\mathrm{Im}\,\mathrm{Tr}\,G/\pi$ is the density of states, which is what
`h.get_dos(mode="Green")` computes

## h.get_hopping_dict()
Return the hoppings as a dictionary `{(n1,n2,n3): matrix}`, indexed by the
lattice vector between the two cells, with `(0,0,0)` the hopping inside the
cell. It is used to give an interaction with a range beyond one cell to the
response functions (see [Spin response and magnons](spin-response.md)), and
to look at or change the hoppings directly. `h.get_multihopping()` returns
the same as a `MultiHopping` object, which can be added and multiplied by
numbers, and `h.set_multihopping()` writes it back

## h.get_rkky()
Compute the RKKY interaction between two magnetic impurities, the effective
exchange $J_{ij}$ mediated by the electrons of `h` (see
[Charge response and RKKY](charge-response.md)).

Optional arguments

- mode="pm": `"pm"` adds two local exchange fields to the Hamiltonian and
  takes the energy difference between parallel and antiparallel alignment,
  $J = -(E_{\uparrow\uparrow}-E_{\uparrow\downarrow})/(4\,\delta J^2)$, with
  $\delta J$ the strength of the two fields, at fixed filling. `"LR"` uses
  linear response, $J_{ij}\propto\chi_{ij}(\omega=0)$, which is faster for
  many separations
- for `mode="pm"`: `ri`, `rj` the positions of the two impurities
  (required), `nk=10`, `dj=1e-1` the strength of the local exchange fields
- for `mode="LR"`: `R=[0,0,0]` the lattice vector between the two cells,
  `ii=0`, `jj=0` the sites in them, `nk=100`, `delta` the broadening (by
  default `1/nk`)

`rkky.rkky_map(h,n=...)` computes it for many separations. On a bipartite
lattice at half filling it is ferromagnetic between sites of the same
sublattice and antiferromagnetic between different sublattices.

## h.get_ldos()
Compute the local density of states at one energy,

$$\rho_i(\omega) = -\frac{1}{\pi}\,\mathrm{Im}\, G_{ii}(\omega+i\delta),$$

on every site. This is what STM measures. See
[Local density of states](ldos.md).

Optional arguments:

- e: the energy $\omega$ (`energy` is also accepted)
- delta=None: broadening $\delta$, 0.001 by default. With `mode="KPM"` it is
  the energy resolution, with `npol=200` polynomials by default
- operator=None: operator the LDOS is projected on, a name, a matrix or an
  `Operator`
- mode="arpack": `"arpack"` (diagonalization on a k-mesh), `"green"`
  (Green's function, 2D only) or `"KPM"` (Chebyshev expansion of each
  site, for large systems, see
  [Kernel polynomial method](kpm.md))
- npol, kpm_radius, scale, kpm_prec: only for `mode="KPM"`, the number of
  polynomials, the radius in hops around each site (`None` for the exact
  result), the energy window of the spectrum, and the precision
- projection="TB": `"TB"` (one value per site), `"TBRS"` (interpolated in
  real space) or `"atomic"` (with atomic orbitals)
- eigmode="complex": non-Hermitian Hamiltonians only, whether `e` is on the
  real or the imaginary axis (see
  [Non-Hermitian Hamiltonians](non-hermitian.md))

Returns the x position, the y position and the LDOS

## h.get_multildos()
Compute the LDOS at many energies. Returns `(x,y,energies,ldos)`, with
`ldos` one row per energy. It also writes one file per energy in the folder
`MULTILDOS/`, with the total DOS in `MULTILDOS/DOS.OUT`, unless
`write=False`. The maps and the DOS have the same normalization as
`h.get_ldos()` and `h.get_dos()`.

Optional arguments:

- energies=linspace(-1,1,100): the energies
- delta, nk: broadening and k-points, as in `h.get_ldos()`; by default
  `delta=0.01`
- mode="diagonalization": `"diagonalization"` or `"KPM"`, with the options of
  `h.get_ldos()` in that mode
- operator=None: operator the LDOS is projected on
- projection="TB": `"TB"` or `"atomic"`

## h.get_chi()
Compute a non-interacting response function between two operators $A$ and
$B$ (density-density by default), $\chi_{AB}(\mathbf q,\omega)$, the
linear response of $\langle A\rangle$ to a perturbation $B$ with momentum
$\mathbf q$ and frequency $\omega$. It is the Lindhard sum over pairs of
states $|n\mathbf k\rangle$ and $|m\mathbf k+\mathbf q\rangle$, each pair
weighted by the difference of their Fermi occupations and by the matrix
elements of $A$ and $B$, with a pole at the energy difference of the pair,
broadened by $\delta$. The result is the response of one site, averaged over
the sites of the unit cell.
See [Charge response and RKKY](charge-response.md).

Optional arguments:

- q=None: the momentum transfer. With `None` the response is averaged over
  all $\mathbf q$ of the mesh, the local response; use `q=[0.,0.,0.]` for
  the uniform response
- A=None, B=None: the two operators (by default the identity, the charge
  response)
- energies, delta, nk: energies, broadening (0.1 by default) and k-points
  per direction (60 by default)

Returns energies and the response function

## h.get_spinchi_ladder()
Compute the transverse spin susceptibility $\chi_{+-}(\mathbf q,\omega)$,
dressed in the random phase approximation (RPA) with the Hubbard $U$ of a
mean-field Hamiltonian,

$$\chi^{RPA} = \chi_0\,(1-U\chi_0)^{-1} .$$

Its poles are the magnons, the spin waves that inelastic neutron scattering
and spin-polarized STM measure. See
[Spin response and magnons](spin-response.md).

Optional arguments:

- q=None, energies, delta, nk: as in `h.get_chi()`. With `q=None` each
  $\mathbf q$ is dressed and then averaged, which gives the local response
- RPA=True: dress the response with the RPA; `False` gives the bare one. An
  interaction between different sites is treated in the pair basis of
  `h.get_transverse_spinchi()`
- T=None: temperature of the occupations, by default equal to `delta`
- chi_prec=None: precision of the sum, `"single"` or `"double"`. By default
  single on the GPU and double on the CPU (see [GPU](gpu.md))

## h.get_rpa_kernel_poles()
Compute the poles of the RPA kernel $1-V(\mathbf q)\chi(\mathbf q,\omega)$,
the frequencies of the collective modes or of the instabilities of the
interacting system.

Optional arguments:

- V=None (required): the interaction. A matrix for an onsite interaction,
  or a dictionary `{(n1,n2,n3): matrix}` (or a `MultiHopping`) for an
  interaction between cells, which is Fourier transformed to
  $V(\mathbf q)$
- A=None, B=None, q=None, energies, delta, nk: as in `h.get_chi()`. A pole
  belongs to one $\mathbf q$, so with `q=None` the kernel is built at
  $\mathbf q=0$ from the averaged response

Returns an `(npoles,2)` array, one row per collective mode, sorted by
frequency: the frequency of the pole and its imaginary part. A small
imaginary part means a sharp mode.

## h.get_magnon_bands()
Compute the magnon bands of a magnetic mean-field state along a q-path.
See [Spin response and magnons](spin-response.md).

Optional arguments:

- method="rpa": `"rpa"`, `"pair"` or `"tdhf"`. `"rpa"` takes the poles of
  the RPA spin response; it works for metals and insulators and needs a
  grid of frequencies. `"pair"` keeps the pair index of the interaction,
  for density-density or exchange interactions between different sites.
  `"tdhf"` solves the time-dependent Hartree-Fock (electron-hole pair)
  eigenvalue problem; it needs no frequency grid, has an exact Goldstone
  mode, and needs a gapped state converged on the same `nk` (or
  `metal=True`)
- qpath=None: the q-path, the default path of the geometry, a list of
  high-symmetry labels, or a list of vectors in reduced coordinates
- nq=None: with a list of vectors, the number of points of the path through
  them, so `qpath=[[0,0,0],[0.5,0,0]], nq=26` is 26 points from 0 to 1/2;
  without it the vectors are the q-points themselves. For the default or a
  labeled path, the density of points, 20 when not given
- energies, delta, nk: as above, for `method="rpa"`
- nk, n, channel, V: for `method="tdhf"`, the k-mesh (the same as in the
  mean-field calculation), the number of branches per q-point, the block
  (`"auto"`, `"spinflip"`, `"all"`), and an interaction instead of `h.V`

Returns `(qs,ws,gammas)` for `method="rpa"`: the index of the q-point along
the path, the frequency of the pole and its imaginary part. With
`method="tdhf"` it returns `(qs,es)`, with `es` the magnon energy; an
imaginary part means that the mean-field state is unstable.

## h.get_transverse_spinchi()
Compute the RPA spin response in the basis of the pairs of sites of the
interaction, instead of single sites. This allows a density-density
interaction between neighbors and an exchange interaction, which the
site-basis RPA cannot include. It works for metals and for non-collinear
states, and gives the full spin- and site-resolved $\chi(\omega)$.

Optional arguments:

- W=None: the interaction, by default the one of the mean-field calculation
- q=None, energies, delta, nk: as in `h.get_chi()`
- component=None: a pair of spin indices `(a,b)` to return only that block
- chi_prec=None: precision of the sum, as in `h.get_spinchi_ladder()`; on the
  CPU only `"double"`

Returns `(energies,chi)`, with `chi` a $3N\times3N$ matrix per frequency,
ordered as $(S_x,S_y,S_z)\times$ site.

## h.get_magnon_energies()
Return the magnon energies at one momentum `Q`, from the spin-flip channel
of the time-dependent Hartree-Fock (Bethe-Salpeter) equation. Same arguments
as `h.get_magnon_bands(method="tdhf")`, with `Q=[qx,qy,qz]` instead of the
q-path. A large imaginary part means that the mean-field state is unstable
against that excitation.

## h.get_goldstone_residual()
Return how far a magnetic mean-field state is from having a zero-energy
magnon at $Q=0$, as required by the Goldstone theorem without spin-orbit
coupling. It is $\|Mv\|/\|v\|$, with $M$ the time-dependent Hartree-Fock
matrix and $v$ a uniform rotation of all spins. It is proportional to the
convergence error of the mean field, so check it before trusting a magnon
dispersion. It also detects a mean field converged on a different `nk`.

Optional arguments:

- nk=10, V=None, channel="auto": as in `h.get_magnon_energies()`
- relative=True: divide by the largest transition energy, to compare models
  with different bandwidths

## h.get_densitychi_RPA()
Compute the RPA charge response

$$\chi^{RPA}(\mathbf q,\omega) = \chi_0\,[1-V(\mathbf q)\chi_0]^{-1}$$

for a density-density interaction with onsite $U$, first, second and third
neighbor $V_1,V_2,V_3$ and a general $V(r)$. The interaction is given here,
not taken from a mean-field calculation. Its poles are plasmons or
charge-density-wave instabilities.

Optional arguments:

- V1=0.0, V2=0.0, V3=0.0, U=0.0, Vr=None: the interaction, as in
  `h.get_mean_field_hamiltonian()`
- rcut=None: range of `Vr`, every pair of sites closer than `rcut`, a site
  with itself included; `None` means 5 in a periodic system and all pairs in
  a finite one
- q=None, energies, delta, nk: as in `h.get_chi()`

## h.get_plasmon_bands()
Compute the plasmon bands, the poles of the RPA charge response along a
q-path. The charge analog of `h.get_magnon_bands()`.

Optional arguments:

- V1=0.0, V2=0.0, V3=0.0, U=0.0, Vr=None: as in `h.get_densitychi_RPA()`
- qpath=None, nq=None, energies, delta, nk: as in `h.get_magnon_bands()`

Returns `(qs,ws,gammas)`, as `h.get_magnon_bands()`.

## h.get_bse()
Solve the Bethe-Salpeter equation for excitons, the bound electron-hole
pairs, on top of a mean-field Hamiltonian, and return a `BSE` object. In
the basis of transitions $v\mathbf k\to c\mathbf k+\mathbf Q$,

$$(\epsilon_{c\mathbf k+\mathbf Q}-\epsilon_{v\mathbf k})A_{vc\mathbf k}
+\sum_{v'c'\mathbf k'}K_{vc\mathbf k,v'c'\mathbf k'}A_{v'c'\mathbf k'} = E_X A_{vc\mathbf k},$$

with $K$ the direct (attractive) minus the exchange (repulsive) kernel. The
excitons below the band gap are the peaks of optical absorption. See
[Excitons](excitons.md).

Optional arguments:

- V=None: the electron-hole interaction. `None` uses `h.V`, the one of the
  mean-field calculation. Otherwise a dictionary `{(n1,n2,n3): matrix}`, or
  a matrix for an onsite interaction; `bsetk.interaction.density_interaction`
  builds one from `U`, `V1`, `V2`, `V3`, `Vr`
- Q=None: total momentum $\mathbf Q$ of the exciton, by default zero
- nk=10: k-points per direction
- nv=None, nc=None: only the `nv` highest valence and `nc` lowest
  conduction bands; `None` takes all
- kernel="full": `"full"`, `"direct"` (only the attraction), `"exchange"`
  (only the exchange, which is the RPA) or `"none"` (free transitions)
- tda=False: Tamm-Dancoff approximation, keeping only the resonant block
- max_memory=2.0: stop, with an error, if the matrix needs more than this
  many GB
- screening=None: the interaction in the direct term. `None` uses the bare
  one, which is time-dependent Hartree-Fock. `"rpa"` uses the static RPA
  screened interaction $W=\varepsilon^{-1}v$ from the bands of `h`.
  `"crpa"` does the same without the transitions inside the `nv`/`nc`
  window. A `ScreenedInteraction` uses one computed before. The exchange
  term always uses the bare interaction. A fitted Hubbard $U$ is already
  screened and should not be screened again
- nkW=None: k-mesh for the screening, by default `nk`; it must be a
  multiple of `nk`
- channel="charge": `"charge"` builds the dielectric matrix on the sites and
  keeps spin rotation symmetry; `"orbital"` uses the full spin-orbital
  matrix. They are the same for a spinless Hamiltonian
- solver="dense": `"dense"` diagonalizes the full matrix, and is the only
  one without the Tamm-Dancoff approximation. `"iterative"` finds the lowest
  excitons without building the matrix, for large k-meshes. `"qtt"` uses a
  tensor-network compression, with a cost that grows as $\log N_k$. The last
  two need `tda=True` (see [Excitons](excitons.md))
- neig=None: number of excitons for `solver="iterative"` (4 by default);
  `"qtt"` only gives one
- gauge="auto": fixes the arbitrary phase of the Bloch states, needed by
  `solver="qtt"`. It does not change any energy

`solver="qtt"` also takes `tolerance` (default `1e-6`), `maxbonddim`,
`maxdim`, `nsweep`, `cutoff`, `coarse_nk` and `unfolding`.

The `BSE` object has `energies`, `amplitudes` (the resonant amplitudes
$A_{vc}(\mathbf k)$), `amplitudesY` (the antiresonant ones, zero with
`tda`), `pairs` (the label `(ik,iv,ic)` of each transition), and the
methods `get_energies(n)`, `get_binding_energies(n)` and
`get_lowest_transition()`. The last one is the lowest free transition, from
which binding energies are measured.

## h.get_exciton_energies()
Return the exciton energies $E_X$, sorted. Same arguments as `h.get_bse()`,
and `n=None` to keep only the `n` lowest.

## h.get_exciton_binding_energies()
Return the binding energies of the excitons, the lowest free transition
minus $E_X$. A positive value means a bound exciton. Same arguments as
`h.get_exciton_energies()`.

## h.get_exciton_states()
Return `(energies,amplitudes)` of the excitons, with the amplitudes
$A_{vc}(\mathbf k)$ indexed by the transition, whose label `(ik,iv,ic)` is
in `pairs.labels` of the `BSE` object. Same arguments as
`h.get_exciton_energies()`.

## h.get_exciton_bands()
Return the exciton band structure $E_X(\mathbf Q)$, one Bethe-Salpeter
calculation per q-point along a path.

Optional arguments:

- qpath=None, nq=None: the q-path and its number of points, as in
  `h.get_magnon_bands()`
- n=None: keep only the `n` lowest excitons at each q-point
- V, nk, nv, nc, kernel, tda, max_memory, screening, nkW: as in
  `h.get_bse()`

Returns `(qs,es)`: the index of the q-point along the path and the exciton
energy. The energy is complex at a q-point where the mean-field state is
unstable. Note that `nv` and `nc` must not cut a degenerate group of bands
(a spinful Hamiltonian without spin-orbit coupling or magnetism needs even
`nv` and `nc`); otherwise a warning is printed.

## h.get_screened_interaction()
Return the static RPA screened interaction
$W(\mathbf q)=\varepsilon^{-1}(\mathbf q)v(\mathbf q)$, with
$\varepsilon = 1 - v\chi_0$ built from the bands of `h`, as a
`ScreenedInteraction` object.

Optional arguments:

- V=None: the bare interaction $v$, as `V` in `h.get_bse()`. `None` uses
  `h.V`; note that a fitted Hubbard $U$ is already screened
- nk=10: k-mesh for the sum over $\mathbf k$ and for the grid of
  $\mathbf q$
- screening="rpa": `"rpa"` (all transitions screen) or `"crpa"` (the
  transitions in `exclude` do not)
- exclude=None: `(vbands,cbands)` that do not screen, needed by `"crpa"`
- channel="charge": `"charge"` or `"orbital"`, as in `h.get_bse()`
- chi_prec=None: precision of $\chi_0$, `"single"` or `"double"`; by default
  single on the GPU and double on the CPU

The object has `qs`, `Wq`, `chi0`, `bare`, `epsmin` (the smallest
eigenvalue of $\varepsilon$ on the mesh), `.at(q)` for the value at a
q-point and `.get_dict()` for the interaction in real space. If an
eigenvalue of $\varepsilon(\mathbf q)$ reaches zero it raises an error: the
mean-field state has a charge or spin instability at that wavevector.

## h.get_polarizability()
Return `(qs,chi0)`, the static polarizability $\chi_0(\mathbf q)$ on the
k-mesh, in the spin-orbital basis, with shape `(nq,norb,norb)`. Takes `nk`,
`exclude` and `chi_prec` as `h.get_screened_interaction()`.

## h.get_fermi_surface()
Compute the spectral weight $A(\mathbf k,\omega)$ on a 2D k-mesh at one
energy. At $\omega=0$ it is the Fermi surface, as seen by ARPES. See
[Spectral functions](spectral-functions.md).

Optional arguments:

- e=0.0: the energy $\omega$
- nk=50: k-points per direction
- delta: broadening
- operator=None: operator that weights the spectral function, for example
  `"sz"`, `"valley"` or `"unfold"`
- primal_mesh=False: use the Brillouin zone of the primitive cell, for a
  supercell built with `store_primal=True`; this is the mesh of an unfolded
  Fermi surface

Returns kx, ky and the spectral weight

## h.get_multi_fermi_surface()
Compute the Fermi surface at many energies, and write one file per energy in
the folder `MULTIFERMISURFACE/`.

Optional arguments:

- energies=[0.0]: the energies
- nk, delta, operator, primal_mesh: as in `h.get_fermi_surface()`

## h.get_surface_kdos()
Compute the spectral function at the surface and in the bulk of a
semi-infinite system, from the surface Green's function. The surface states
of a topological insulator appear here as bands inside the bulk gap. See
[Spectral functions](spectral-functions.md).

Optional arguments:

- kpath: the k-path (generated if not given)
- energies, delta: energies and broadening

Returns k, energy, surface spectral weight and bulk spectral weight; also
writes `KDOS.OUT`

## h.get_qpi()
Compute the quasiparticle interference pattern, the Fourier transform of
the LDOS modulation around a defect, as measured by Fourier-transform STM
(2D systems only). See [Quasiparticle interference](qpi.md). Returns
`(q,energies,qpi)`, with `qpi` one row per energy and one column per
wavevector, and also writes the maps to `MULTIQPI/` and the DOS to
`DOS.OUT`.

Optional arguments:

- energies, nk, delta: as above
- mode="response": `"pm"` computes the autocorrelation of the
  momentum-resolved spectral function, the QPI of a real scatterer;
  `"response"` uses a joint density of states of the clean bands, which is
  faster; `"tmatrix"` computes a single impurity in the infinite system
  with the T-matrix, and `"born"` the same impurity to first order
- impurity=None: the impurity of `"tmatrix"` and `"born"`, a number (onsite
  potential on the first site of the cell), one onsite potential per site,
  or a matrix with the dimension of the cell
- nunfold=1: unfold the QPI of a supercell to the Brillouin zone of the
  primitive cell, given as the linear size of the supercell (`np.sqrt(3)`
  for the $\sqrt3\times\sqrt3$ one)
- write=True: with `False` no file is written

## h.get_qpi_impurity()
Compute the quasiparticle interference with real impurities: they are put
in a supercell, the LDOS is computed in real space, and it is Fourier
transformed (2D systems only). Returns `(r,ldos_r,q,qpi_q)`.

Optional arguments:

- nsuper=10: size of the supercell (a number or `(n1,n2)`)
- impurities=[]: list of dictionaries, each
  `{"position"|"index": ..., "onsite": v}` or
  `{"position"|"index": ..., "vacancy": True}`
- energies=0.0, delta, nk: as above
- num_waves=20: initial number of eigenstates near each energy; it is
  increased automatically when needed, so a small value only costs time
- write=True, output_folder="QPI_IMPURITY": also write the maps to files

## h.get_spin_splitting_density()
Compute the spin splitting of a collinear magnet as a function of energy.
Each pair of spin-up and spin-down bands contributes its squared splitting
$(\epsilon_{n\uparrow}-\epsilon_{n\downarrow})^2$, broadened by a
Lorentzian at the mean energy of the pair, so the integral over energy is
the squared splitting summed over bands and averaged over the Brillouin
zone. It is the quantity that shows an altermagnet, spin-split bands without
net magnetization. Returns `(energies,values)`. See
[Altermagnets](altermagnets.md).

Optional arguments:

- nk=20: k-points per direction
- energies: the energies (by default 400 points from -3 to 3)
- delta=1e-2: broadening
- tol=1e-7: largest spin-mixing matrix element accepted; above it the
  method raises an error, since the spin splitting is not defined with
  spin-orbit coupling or non-collinear order

## h.get_average_spin_splitting()
Return one number: the spin splitting of a typical band, averaged over the
bands and over the Brillouin zone. It does not depend on the size of the
unit cell used to describe the crystal.

Optional arguments:

- nk=20: k-points per direction
- tol=1e-7: as above

Use `h.get_spin_splitting_vs_energy()` to get the largest splitting instead
of the typical one.

## h.get_spin_splitting_vs_energy()
Compute the largest spin splitting $|\epsilon_{n\uparrow}-\epsilon_{n\downarrow}|$
over the Brillouin zone, as a function of the mean energy of the pair.
Returns `(energies,values)`, in the same form as
`h.get_spin_splitting_density()`.

Optional arguments:

- nk=100: k-points per direction
- energies=None: centers of the energy bins; otherwise `nbins` points from
  `emin` to `emax`
- nbins=400, emin=None, emax=None: used without `energies`; by default the
  range of the energies found
- tol=1e-7: as above

Empty bins are `0.0`.

## h.get_spin_chern()
Return the spin Chern number $(C_+-C_-)/2$ of a two-dimensional spinful
insulator. The occupied states are split by the sign of the projected spin
$P s_z P$, so the result stays quantized when $s_z$ is not conserved, for
example with Rashba coupling. See
[Topological invariants](topological-invariants.md).

Optional arguments:
- operator="sz": the spin component, a name, a matrix or an `Operator`
- nk=40: k-points per direction
- write=True: with `False` nothing is written to `CHERN.OUT` and
  `BERRY_CURVATURE.OUT`

It raises `ValueError` if the projected spin has no gap or the system is a
metal.

## h.get_mirror_chern()
Return the mirror Chern number $(C_{+i}-C_{-i})/2$ of a two-dimensional
insulator with the mirror symmetry $z\to -z$, with $C_{\pm i}$ the Chern
numbers of the two mirror sectors. The mirror is found from the geometry and
the Hamiltonian; without it the method raises `ValueError`.

Optional arguments:
- nk=40: k-points per direction

## h.get_winding_number()
Return the winding number of a one-dimensional Hamiltonian with chiral
symmetry,

$$W = \frac{1}{2\pi i}\int dk\,\partial_k \log\det q(k),$$

with $q(k)$ the off-diagonal block of $H(k)$ in the basis where the chiral
operator is diagonal. It is the number of zero modes at each end of an open
chain. See [Topological invariants](topological-invariants.md).

Optional arguments:
- chiral=None: the chiral operator, a name, a matrix or an `Operator`; by
  default the sublattice operator of a normal Hamiltonian and
  $\sigma_y\tau_y$ on each site of a Nambu one. It raises `ValueError` if it
  does not anticommute with the Hamiltonian
- nk=200: k-points in the Brillouin zone

## h.get_quadrupole_moment()
Return the quadrupole moment $q_{xy}$, 0 or 1/2, of a two-dimensional
insulator with the mirrors $M_x$ and $M_y$, from the nested Wilson loop. A
value 1/2 means corner charges $\pm e/2$, a higher-order topological
insulator. See [Topological invariants](topological-invariants.md).

Optional arguments:
- nk=40: k-points per direction
- nocc=None: number of occupied bands, counted from the lowest; by default
  the bands below zero energy
- tol=1e-4: smallest distance of the Wannier bands from 0 and 1/2; below it
  the Wannier gap is closed and the method raises `ValueError`

## h.get_wannier_sector_polarization()
Return the polarization of one sector of the Wannier bands, along the
direction perpendicular to the first Wilson loop, from the nested Wilson
loop. It is given modulo 1, in $[-1/4,3/4)$, so that 0 and 1/2 are not at
the edge of the interval.

Optional arguments:
- loop=0: direction of the first Wilson loop
- sector="+": the Wannier bands with $0<\nu<1/2$, or `"-"` for
  $-1/2<\nu<0$
- gauge="lattice": all orbitals at the origin of the cell, or `"atomic"`,
  each orbital at its position
- nk=40, nocc=None, tol=1e-4: as in `h.get_quadrupole_moment()`

## h.get_chern()
Return the Chern number of the occupied bands,

$$C = \frac{1}{2\pi}\int_{BZ} \Omega(\mathbf k)\, d^2k ,$$

with $\Omega$ the Berry curvature. It is the quantized Hall conductivity
$\sigma_{xy} = C\,e^2/h$, and the number of chiral edge states. See
[Topological invariants](topological-invariants.md).

Optional arguments:
- nk: k-points per direction, 10 with `integration="grid"` and 20 with
  `integration="qtci"`; with `integration="wannier"`, k-points along each
  Wilson loop, 30 by default
- integration="grid": how the integral is done. `"grid"` sums the Berry
  curvature on an `nk` x `nk` mesh, and the result is always an integer.
  `"qtci"` uses tensor cross interpolation, which samples adaptively; it is
  accurate for a smooth curvature (a large gap) but not for a sharp peak.
  `"wannier"` counts the winding of the hybrid Wannier centers
- nt=100: for `integration="wannier"`, the number of momenta in the other
  direction; a small gap needs more
- nocc=None: for `integration="wannier"`, the number of occupied bands; by
  default the bands below zero energy
- operator=None: an operator that restricts the Berry curvature, as in
  `h.get_berry_curvature()`
- write=True: also write the Chern number to `CHERN.OUT` and the Berry
  curvature to `BERRY_CURVATURE.OUT`

## h.get_berry_curvature()
Return the Berry curvature of the occupied bands,

$$\Omega(\mathbf k) = i\sum_{n\in\mathrm{occ}}\left(\langle\partial_{k_x}u_n|\partial_{k_y}u_n\rangle - \langle\partial_{k_y}u_n|\partial_{k_x}u_n\rangle\right),$$

as a map over the Brillouin zone, `(kx,ky,berry)`, three flat arrays. This
is the curvature that `h.get_chern()` integrates.

In two dimensions the map covers a whole number of Brillouin zones, and the curvature
is per unit area of the reduced zone, whose area is one. So the Chern number
is `np.mean(berry)/(2*np.pi)`, the same as `h.get_chern()` when `nk`
resolves the curvature.

Optional arguments:

- nk=100: k-points per direction (the map has `nk*nk` points, and
  `berry.reshape(nk,nk)` has `kx` along the first index)
- reciprocal=True: `kx,ky` in Cartesian coordinates, to plot a hexagonal
  Brillouin zone without distortion. The map covers a rectangle centered at
  $\Gamma$ made of whole zones: one zone for a square lattice, a square of
  side $|\mathbf b_1|$; two zones for a hexagonal lattice, of sides
  $|\mathbf b_1|$ and $\sqrt{3}|\mathbf b_1|$. A lattice without such a
  rectangle (an oblique one) gets the parallelogram of its two reciprocal
  lattice vectors, one zone, whose points are not on a rectangular grid.
  With `reciprocal=False`, `kx,ky` are reduced coordinates from $-1$ to $1$,
  four zones
- mode="Wilson": `"Wilson"` uses the Fukui-Hatsugai-Suzuki plaquettes;
  `"Green"` uses the Green's-function Kubo formula, and is used
  automatically with `operator`
- operator=None: restrict the curvature to a subspace, for example
  `"valley"` for the valley-resolved curvature (see [Operators](operators.md))
- nsuper=1: number of copies of the window in each direction
- kpath: compute along a k-path instead of on a 2D mesh; for a curvature
  to plot next to the bands, `h.get_berry_curvature_path()` is simpler
- delta=0.001: broadening for `mode="Green"`
- write=True: also write the map to `BERRY_MAP.OUT`

## h.get_berry_curvature_path()
Return the Berry curvature of the occupied bands along a k-path, the same
$\Omega(\mathbf k)$ as `h.get_berry_curvature()`, as `(index,berry)`: the
index of each k-point of the path and the curvature there. It is the
curvature to plot under a band structure, where its peaks show the k-points
of small gaps.

Optional arguments:

- kpath=None: the k-path, a list of k-points or of high-symmetry labels; by
  default the k-path of `h.get_bands()`
- nk=600: number of k-points of the path
- reciprocal=False: with `True`, the k-points of `kpath` are Cartesian, in
  the coordinates of `h.get_berry_curvature()`, instead of reduced
- dk=0.01: side of the plaquette of the Wilson loop, in reduced coordinates
- mode="Wilson", operator=None, delta=0.001: as in `h.get_berry_curvature()`
- write=True: also write the k-points and the curvature to
  `BERRY_CURVATURE.OUT`

## h.get_quantum_geometric_tensor()
Return the quantum geometric tensor of a set of bands $S$ at one k-point,

$$Q_{\mu\nu}(\mathbf k) = \sum_{n\in S}\langle\partial_\mu u_n|\left(1-P_S\right)|\partial_\nu u_n\rangle ,$$

with $P_S$ the projector on the bands. Its real part is the quantum metric
and its imaginary part is $-\Omega/2$, with $\Omega$ the Berry curvature.
See [Quantum geometry](quantum-geometry.md).

Optional arguments:
- k=[0.,0.,0.]: the k-point
- occ_idxs=None: the band indices of $S$; by default the bands with
  $E<0$, as in `h.get_chern()`
- non_abelian=False: with `True`, return the full tensor in the orbital
  basis, $\sum_{m,n\in S}|u_m\rangle Q^{mn}\langle u_n|$, a matrix for each
  pair of directions, which does not depend on the choice of basis inside a
  degenerate subspace
- degeneracy_tol=1e-8: if a band of $S$ is degenerate with a band outside
  it within this tolerance, the method raises `ValueError`
- gauge="atomic": `"atomic"` uses the positions of the orbitals in the
  Bloch phase, the physical quantum geometry; `"lattice"` puts every orbital
  at the origin of its cell
- coordinates="reduced": derivatives along the reduced momenta, or
  `"cartesian"` along $k_x$, $k_y$, $k_z$

## h.get_quantum_metric()
Return the quantum metric $g_{\mu\nu}=\mathrm{Re}\,Q_{\mu\nu}$, with the same
arguments as `h.get_quantum_geometric_tensor()`.

## h.get_dvector_non_unitarity()
Return the non-unitarity vector of a triplet superconductor,
$\vec q = i(\vec d \times \vec d^*)$, on each site, for a BdG Hamiltonian
(see [Superconductivity](superconductivity.md)). It is real, zero for a
unitary state, and otherwise is the spin of the Cooper pairs: parallel to
the magnetization in a ferromagnetic triplet superconductor, and along $+z$
for pure $\Delta_{\uparrow\uparrow}$ pairing.

Optional arguments

- nk = 10: k-points per direction

Returns an array of shape (number of sites, 3).

## h.write_non_unitarity()
Write $\vec q$ as a map in real space to `NON_UNITARITY_MAP.OUT`, with
columns $(x,y,z,q_x,q_y,q_z)$.

Optional arguments

- nrep = 2: number of copies of the unit cell in each direction
- nk = 10: k-points per direction

## h.get_average_dvector()
Return the squared components $(|d_x|^2,|d_y|^2,|d_z|^2)$ of the d-vector of
a BdG Hamiltonian, averaged over $\mathbf k$.

Optional arguments

- nk = 10: k-points per direction
- spatial_sum = True: average over the sites, returning one 3-vector
- non_unitarity = False: average the squared components of $\vec q$
  instead. This only says whether the state is non-unitary, not the
  direction; use `h.get_dvector_non_unitarity()` for the vector.

## h.get_superfluid_weight()
Return the superfluid weight tensor $D_s^{ab}$ of a BdG Hamiltonian in one,
two or three dimensions, the response of the supercurrent to a phase
gradient,

$$D_s^{ab} = \frac{\partial^2\Omega(\mathbf A)}{\partial A_a\,\partial A_b}\Big|_{\mathbf A=0},$$

with $\Omega$ the grand potential and $\mathbf A$ a vector potential. It sets
the London penetration depth, $\lambda^{-2}\propto D_s$. See
[Superconductivity](superconductivity.md).

Optional arguments:

- nk=20: k-points per direction
- T=0.0: temperature
- mode="kubo": `"kubo"` uses the multiband Kubo formula, and
  `"finite_difference"` the numerical second derivative of the grand
  potential
- gauge="atomic": `"atomic"` uses the full bond vector in the phase (the
  physical Peierls substitution), `"lattice"` only the lattice vector
- decompose=False: with `True`, return a dictionary with `"total"`,
  `"conventional"`, `"geometric"`, `"delta"` and `"gauge"`; the conventional
  part comes from the band dispersion and the geometric part from the
  quantum metric

Returns a `(dim,dim)` array, or a dictionary with `decompose=True`.
`decompose=True` raises `ValueError` when the decomposition is not defined,
and with `mode="finite_difference"`.

## h.get_bkt_temperature()
Return the Berezinskii-Kosterlitz-Thouless temperature of a 2D BdG
Hamiltonian, from the Nelson-Kosterlitz condition
$T_{\rm BKT} = (\pi/8)D_s(T_{\rm BKT})$, with $|\Delta|$ kept fixed. Arguments
`nk=20`, `tmax=None`, `tol=1e-6`, `maxite=60`, `maxexpand=20`, and `gauge`.
`tmax` is the upper end of the search, $(\pi/8)D_s(0)$ by default; it is
doubled up to `maxexpand` times if needed, and if no crossing is found the
method raises `ValueError`. Returns a number.

## h.get_nonlinear_drude_conductivity()
Compute the $l$-th order nonlinear Drude conductivity
$\sigma^{x^{l_1}y^{l_2};b}$ of a collinear magnet, the current along $b$
of order $E^l$ in the electric field, for spin $s$,

$$\sigma_s^{x^{l_1}y^{l_2};b} = \frac{(-e/\hbar)^{l+1}}{(i\omega+1/\tau)^l}
\int\frac{d^Dk}{(2\pi)^D}\, f^{(0)}_s\,
\frac{\partial^{l+1}\epsilon_s}{\partial k_x^{l_1}\partial k_y^{l_2}\partial k_b},$$

with $l=l_1+l_2$, $f^{(0)}_s$ the Fermi function and $\epsilon_s$ the bands of
spin $s$. The lowest order that is not zero
gives the wave index of an altermagnet ($d$: 1, $f$: 2, $g$: 3, $i$: 5; a
$p$-wave magnet has none, and $l=0$ is always zero). Spin must be
conserved. See [Altermagnets](altermagnets.md).

Optional arguments

- field = "x": one character per power of the electric field, for example
  "yyyyy" for the fifth-order response to $E_y$
- current = "x": direction of the current
- channel = "spin": "spin", "charge", "up" or "dn"
- nk = 100: k-points per direction
- T = 0.01: temperature of the occupations
- mu = 0.0: chemical potential
- tau = 1.0: relaxation time $\tau$; the $l$-th order scales as $\tau^l$
- omega = 0.0: frequency, which enters as $1/(i\omega + 1/\tau)^l$
- degeneracy_tol = 1e-8: relative band spacing below which bands are taken
  as degenerate

Returns a complex number

## h.get_nonlinear_drude_components()
All components of the $l$-th order nonlinear Drude conductivity, as a
dictionary `{"x^l1 y^l2;b": value}`. It uses one k-mesh for all components,
so it is faster than separate calls.

## h.get_nonlinear_drude_orders()
For each order $l = 0$ to `lmax`, the largest
$|\sigma^{x^{l_1}y^{l_2};b}|$ over the components of that order. The lowest
$l$ with a nonzero value is the wave index.

Optional arguments

- lmax = 6: highest order
- every optional argument of `h.get_nonlinear_drude_conductivity()`

Returns a list of numbers

## h.get_optical_conductivity()
Compute the optical conductivity tensor $\sigma_{ab}(\omega)$ with the
Kubo-Greenwood formula, with an intraband (Drude) and an interband part.
$\mathrm{Re}\,\sigma_{aa}$ is the optical absorption, and the DC limit of
$\sigma_{xy}$ is the Hall conductivity. See
[Optical conductivity](optical-conductivity.md).

Optional arguments:

- energies=None: the frequencies $\omega$
- nk=20: k-points per direction
- T=None: temperature (by default equal to `delta`)
- delta=0.1: broadening $\eta$
- intraband=True, interband=True: include each of the two parts
- component=None: for example `"xy"`, to return only that component
- degeneracy_tol=1e-6: relative tolerance to treat two bands as degenerate

Returns `(energies,sigma)`, with `sigma` complex of shape `(nw,3,3)`, or
`(nw,)` with `component`. The unit is $e^2/\hbar$, so $e^2/h$ is
$1/(2\pi)$. 3D and Nambu Hamiltonians raise `NotImplementedError`.

## h.get_shg_conductivity()
Compute the second-harmonic conductivity $\sigma_{abc}(2\omega;\omega,\omega)$,
the current at $2\omega$ driven by a field at $\omega$,
$J_a(2\omega)=\sigma_{abc}E_b(\omega)E_c(\omega)$, symmetric in $b$ and $c$.
It is zero in a crystal with inversion symmetry. See
[Second-harmonic generation](optical-conductivity.md#second-harmonic-generation).

Optional arguments:

- energies=None: the fundamental frequencies $\omega$
- nk=20: k-points per direction
- T=None: temperature (by default equal to `delta`)
- delta=0.1: broadening $\eta$ of each frequency
- components=None: one component such as `"xxx"`, a list of them, or all 27
- gauge="velocity": `"velocity"`, or `"length"` as a check for an insulator
  at `T=0` with no degenerate bands
- degeneracy_tol=1e-6: relative tolerance to treat two bands as degenerate in
  the length gauge

Returns `(energies,sigma)`, with `sigma` a complex array of length `nw` for one
component and a dictionary of such arrays for a list. The unit is
$e^3a^{3-d}/(\hbar t)$. 3D and Nambu Hamiltonians raise `NotImplementedError`.

## h.get_shg_susceptibility()
Return the second-harmonic susceptibility
$\chi_{abc}=i\sigma_{abc}/(2(\omega+i\eta)\epsilon_0)$ in SI units, nm$^2$/V
for a 2D system (the sheet susceptibility). Arguments `length_unit=1.0`, the
length unit of the geometry in Å, `energy_unit=1.0`, the energy unit of the
Hamiltonian in eV, and every argument of `h.get_shg_conductivity()`. Returns
`(energies,chi)` in the same layout.

## h.get_drude_weight()
Return the Drude weight tensor, the weight of the intraband
$\delta(\omega)$ peak of $\mathrm{Re}\,\sigma(\omega)$. Arguments `nk=20`,
`T=0.05`, `degeneracy_tol=1e-6`. Returns a real `(3,3)` array.

## h.get_sum_rule_weight()
Return the diamagnetic weight $W$ of the optical f-sum rule,
$\int \mathrm{Re}\,\sigma_{aa}(\omega)\,d\omega = \pi W_{aa}$. Arguments
`nk=20`, `T=0.05`. Returns a real symmetric `(3,3)` array.

## h.get_entanglement_entropy()
Return the entanglement entropy of a region $A$ of the ground state,

$$S = -\sum_n\left[\zeta_n\log\zeta_n + (1-\zeta_n)\log(1-\zeta_n)\right],$$

with $\zeta_n$ the eigenvalues of the correlation matrix
$C_{ij}=\langle c^\dagger_i c_j\rangle$ restricted to $A$. See
[Entanglement](entanglement.md).

Optional arguments:

- region=None: the region, a list of site indices, a boolean mask, a
  function of the position, or a fraction of the cells; `None` takes half
  of the system
- nsuper=10: number of unit cells in the ring that is cut (periodic
  Hamiltonians)
- direction=None: lattice direction perpendicular to the cut, by default
  the last periodic one
- kpar=None: momentum parallel to the cut; with `None` a 2D Hamiltonian is
  averaged over `nk` momenta
- nk=20: number of momenta of that average
- fermi=0.0: the states with $E<$ `fermi` are occupied (must be 0 for BdG)

Returns a number. A state exactly at the Fermi energy raises an error,
since the ground state is then not unique.

## h.get_entanglement_spectrum()
Return the eigenvalues of the entanglement Hamiltonian,
$\xi_n = \ln[(1-\zeta_n)/\zeta_n]$, of a region. Same arguments as
`h.get_entanglement_entropy()` (with `nk=41` by default).

Returns a sorted array of $\xi_n$ for a 0D or 1D Hamiltonian or a single
`kpar`. For a 2D Hamiltonian with `kpar=None` it returns `(ks,xis)`, with
`xis` of shape `(nk,nA)`, the Li-Haldane entanglement spectrum as a function
of the parallel momentum. In a topological phase it shows the same spectral
flow as the edge states.

## h.get_wannier_hamiltonian()
Build maximally localized Wannier functions for a set of bands, and return
the tight-binding Hamiltonian in this basis. See
[Wannierization](wannierization.md).

Arguments:

- bands = [a,b]: first and last band (counted from 0, both included)

Optional arguments:

- nk=12: k-points per direction
- cutoff=1e-6: hoppings smaller than this are dropped (the hopping inside the
  cell is always kept). With `0.0` the bands are reproduced exactly
- symmetries=None: `"auto"` to find and impose the point group, or a list of
  `symmetrytk.pointgroup.SymmetryOperation`
- num_wann=None: number of Wannier functions. `None` uses the whole set of
  bands, which is then reproduced exactly. A smaller number uses the
  Souza-Marzari-Vanderbilt disentanglement, and then only the frozen window
  is reproduced exactly. A value outside `1..len(bands)` raises `ValueError`
- dis_win_min, dis_win_max=None: outer energy window, the states that can
  enter the Wannier functions at each k-point
- dis_froz_min, dis_froz_max=None: frozen window, whose states are
  reproduced exactly. `dis_froz_max` alone is enough; `dis_froz_min` alone
  raises `ValueError`
- dis_num_iter=200: maximum number of iterations of the disentanglement; if
  it does not converge, a warning is printed

A `dis_*` argument without a smaller `num_wann` raises `ValueError`.
Disentanglement with a BdG Hamiltonian, with `symmetries=`, or with
`auto_split_clusters=True` raises `NotImplementedError`.

Returns a new, smaller Hamiltonian. `.wannier_centres`, `.wannier_spreads`
and `.wannier_spread_total` give the centers and spreads of the Wannier
functions, and `.wannier_num_wann` and `.wannier_disentanglement_window`
what was used (the window is `None` without disentanglement)

## h.get_szsz_mean_field_hamiltonian()
Self-consistent Hartree-Fock mean field of an Ising exchange interaction
$\sum_{ij} J_{ij} S^z_iS^z_j$ (see [Exchange interactions](exchange.md)).
$J>0$ is antiferromagnetic and $J<0$ ferromagnetic.

Optional arguments:

- J1, J2, J3 = 0.: first, second and third neighbor couplings
- Jr=None: a coupling as a function of the distance, as `Vr` in
  `h.get_mean_field_hamiltonian()`; `Jr(0)` is an onsite term
  $J_r(0)S^z_iS^z_i/2$, so a coupling between bonds must be zero at zero
  distance
- filling, mf, nk, maxerror, mix, constrains: as in
  `h.get_mean_field_hamiltonian()`
- return_total_energy=False: also return the total energy

It also works for BdG Hamiltonians, with both the normal and the pairing
channels.

Returns the self-consistent Hamiltonian (it raises an error if the loop does not converge)

## h.get_sxsx_mean_field_hamiltonian() / h.get_sysy_mean_field_hamiltonian()
The same as `h.get_szsz_mean_field_hamiltonian()`, for an $S^x_iS^x_j$ or
$S^y_iS^y_j$ interaction. Also works for BdG Hamiltonians.

## h.get_exchange_mean_field_hamiltonian()
Self-consistent mean field of an anisotropic exchange
$\sum_{ij}\left(J^x_{ij} S^x_iS^x_j + J^y_{ij}S^y_iS^y_j + J^z_{ij}S^z_iS^z_j\right)$.

Optional arguments:

- Jx1, Jx2, Jx3, Jy1, Jy2, Jy3, Jz1, Jz2, Jz3 = 0.: first, second and third
  neighbor couplings for each spin component
- Jxr, Jyr, Jzr=None: couplings as a function of the distance, one per
  component
- mf, filling, nk, maxerror, mix, constrains: as above

It also works for BdG Hamiltonians, with the normal and the pairing
channels. An exchange interaction can give superconductivity: for example
an antiferromagnetic $J$ alone, started from a random guess, can converge to
a superconducting state.

Returns the self-consistent Hamiltonian (it raises an error if the loop does not converge)

## h.get_combined_mean_field_hamiltonian()
Self-consistent mean field of density-density interactions (onsite $U$,
neighbor $V_1$, $V_2$, $V_3$, and $V_r$) together with a Heisenberg
exchange $\sum_{ij}J_{ij}\vec S_i\cdot\vec S_j$. See
[Mean-field Hubbard models](mean-field.md) and
[Exchange interactions](exchange.md).

Optional arguments:

- U, V1, V2, V3, Vr: as in `h.get_mean_field_hamiltonian()`
- J1, J2, J3 = 0.: Heisenberg exchange for the first, second and third
  neighbors
- Jr=None: an exchange $J_r(\mathbf r_1,\mathbf r_2)$ as a function of the
  positions, as `Vr`. A site with itself is included, so `Jr(0)` is an
  onsite term $J_r(0)\,\vec S_i\cdot\vec S_i/2$, which at the mean-field
  level is a Hubbard $U=-3J_r(0)/4$; an exchange between bonds must be zero
  at zero distance
- rcut=None: range of `Vr` and `Jr`, every pair of sites closer than
  `rcut`, a site with itself included (`Vr(0)` is an onsite $U=V_r(0)$);
  `None` means 5 in a periodic system and all pairs in a finite one
- J1x, J1y, J1z = 0.: anisotropic correction to $J_1$, only for first
  neighbors (the first-neighbor $J_z$ is $J_1+J_{1z}$)
- mf, filling, nk, maxerror, mix, constrains: as above
- integration="ed": the density matrix of each iteration is computed by
  exact diagonalization. `"kpm"` uses a Chebyshev (KPM) expansion instead,
  for systems too large to diagonalize. With `is_sparse=True` in
  `g.get_hamiltonian()` the memory grows linearly with the number of sites.
  A Nambu Hamiltonian also works, sparse or dense, and a dense one comes back dense. With
  `"kpm"`: `scale` is the energy window (found automatically if not given;
  if given, it must contain the whole spectrum, or it raises `ValueError`),
  `npol` the number of Chebyshev moments, `ne` the number of energies in the
  occupied window, `cores` the number of parallel workers, and `kpm_radius`
  the number of hops around each site that are kept (`None` for the whole
  system). A finite `kpm_radius` makes the time linear in the number of
  sites, and it is accurate for a gapped state or at finite temperature
  (see [Kernel polynomial method](kpm.md))
- `use_jax=True, solver=...`: find the self-consistent solution with a
  nonlinear solver instead of simple mixing. `"newton"` is the default and
  `"error_gradient"` the most robust. They can stop before `maxerror`, and
  then the method raises an error as for any calculation that does not
  converge. Only for a normal (non-BdG) Hamiltonian, with exact
  diagonalization and without `constrains`:

```python
from pyqula import geometry
h = geometry.chain().get_hamiltonian()
hmf = h.get_combined_mean_field_hamiltonian(U=4.0,J1=-0.5,filling=0.5,
        use_jax=True,solver="newton") # JAX-derivative-based SCF solver
```

For a BdG Hamiltonian all the channels include the normal and the pairing
terms, so an exchange alone can give superconductivity and not only
magnetism (see [Exchange interactions](exchange.md)).

Returns the self-consistent Hamiltonian (it raises an error if the loop does not converge)

`filling` can also be an array with one value per site, with the same
meaning as the number (the fraction of the two states of the site that is
occupied). Then $\langle n_i\rangle=$ `filling[i]` is imposed on each site,
with one Lagrange multiplier (a local chemical potential) per site found in
the same self-consistent loop. The converged potentials are in the `fermi`
attribute of the result. If the mean field or the constraint does not
converge, the method raises an error. Only for a normal (non-BdG) Hamiltonian with
`mu=None`, with `integration="ed"` or `"kpm"`. `SpinonHamiltonian` uses this to impose
exactly one fermion per site.

## SpinonHamiltonian(g)
Abrikosov pseudofermion mean-field Hamiltonian of a spin-1/2 Heisenberg
model on the geometry `g`. Each spin is written as
$\vec S_i = \frac12 f^\dagger_i\vec\sigma f_i$ with one fermion per site, and
the exchange is decoupled in the hopping (RVB) and magnetic channels. See
[Pseudofermion mean field](pseudofermions.md). It is imported with
`from pyqula.spinon import SpinonHamiltonian`; it has no bare hopping, and
the couplings are given to `h.get_mean_field_hamiltonian()` as `J1`, `J2`,
`J3`, `Jr`, `J1x`, `J1y`, `J1z`.

- `h.get_mean_field_hamiltonian(J1=...,nk=...,...)`: the same arguments as
  `h.get_combined_mean_field_hamiltonian()`, except `filling`, since there
  is always exactly one fermion per site. Returns the self-consistent
  Hamiltonian (it raises an error if the loop does not converge), with two more attributes:
  - `h2.local_occupation`: $\langle n_i\rangle$ on each site (between 0 and
    2, it should be 1.0)
  - `h2.constraint_lambda`: the Lagrange multiplier of each site (a local
    chemical potential)

## KondoLatticeHamiltonian(hc)
Abrikosov pseudofermion (Read-Newns) mean-field Hamiltonian of a Kondo
lattice, built from the conduction-electron Hamiltonian `hc`. A localized
$f$ site, with no bare hopping, is added next to each conduction site, and
the Kondo coupling $J\,\vec s_i\cdot\vec S_i$ is decoupled in a hybridization
$V_i$ between $c$ and $f$ electrons. A finite $V_i$ is the Kondo (heavy
Fermi liquid) state. See [Pseudofermion mean field](pseudofermions.md).
It is imported with `from pyqula.kondolattice import KondoLatticeHamiltonian`,
and $J$ is given to `h.get_mean_field_hamiltonian()` as `J`.

- `h.get_mean_field_hamiltonian(J=...,filling=...,mf=(V,lam),nk=...,...)`:
  finds the hybridization and the Lagrange multiplier of the constraint
  self-consistently. Returns the self-consistent Hamiltonian (it raises an
  error if the loop does not converge), with three more attributes:
  - `h2.local_occupation`: $\langle n_f\rangle$ on each localized site (it
    should be 1.0)
  - `h2.hybridization`: $V_j$ on each localized site
  - `h2.constraint_lambda`: the Lagrange multiplier of each site

## GrapheneGeometry(g)
A `Geometry` for graphene multilayers `g` (bilayer, twisted bilayer,
twisted trilayer...), with a method `.relax()` (see
[Structural relaxation](relaxation.md)). It is imported with
`from pyqula.graphenetk.geometry import GrapheneGeometry`. `g` must have
`has_sublattice=True`, otherwise it raises `ValueError`.

- `g.relax(nrep=1,maxiter=500,verbose=False,layer_pairs=None,gsfe_coeffs=...,elastic_coeffs=...)`:
  minimizes the stacking energy between layers (the generalized stacking
  fault energy, GSFE) plus the elastic energy inside each layer, with
  respect to in-plane displacements, and returns a new relaxed
  `GrapheneGeometry`. `layer_pairs` gives the pairs of layers (ordered by
  $z$) with a stacking energy, by default all neighboring pairs;
  `gsfe_coeffs` and `elastic_coeffs` replace the graphene parameters.

## GrapheneHamiltonian(geometry)
A Hamiltonian for a graphene multilayer `geometry` (usually a
`GrapheneGeometry`, relaxed or not; see
[Structural relaxation](relaxation.md)). It is imported with
`from pyqula.graphenetk.hamiltonian import GrapheneHamiltonian`. The
hoppings decay with the distance, as in
`specialhopping.twisted_matrix` (`ti=0.12,lambi=8.0,lamb=12.0,dl=3.0`, the
same as `specialhamiltonian.twisted_bilayer_graphene`), so the relaxed
positions change the electronic structure. `mgenerator=...` gives another
hopping function.

## h.get_central_heterostructure()
Build a two-terminal `Heterostructure` with `h`, a finite (0D) Hamiltonian,
as the central region, connected to two semi-infinite chains at sites `i`
and `j` (see [Quantum transport](transport.md)).

Optional arguments:

- i=0, j=None: the sites where the leads are attached (counted from 0); by
  default `j` is the last site
- left=None, right=None: the Hamiltonians of the leads; by default a
  spinless `geometry.chain()`. Add pairing (`add_swave`) to one of them, or
  to `h`, for a normal-superconductor junction. Only one of `h`, `left` and
  `right` can have pairing

Returns a `Heterostructure`, so `didv`, `get_dos`, `get_kappa` and the
other methods work. `landauer` raises `NotImplementedError` when there is
pairing; use `didv` then. A central region that is not 0D raises
`NotImplementedError`.
