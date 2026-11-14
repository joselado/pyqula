# Chebyshev methods from the papers: what pyqula has, what was built, what is open

On 14 November 2026 we surveyed the Chebyshev (KPM) methods for spectral and transport
calculations on large lattices, and compared each with what pyqula has. The methods come
from these papers, each read from its TeX source:

- Garcia, Covaci and Rappoport, PRL 114, 116602, arXiv:1410.8140: the Kubo-Bastin
  conductivity from double Chebyshev moments (read in an earlier session, see
  `kpm_optical_conductivity.md`).
- Ferreira and Mucciolo, PRL 115, 106601, arXiv:1507.00488: the exact Chebyshev series of
  the resolvent at complex energy, and the single-shot Kubo-Greenwood conductivity,
  sigma(E) = (2 hbar e^2/(pi Omega)) Tr[Im G v Im G v], from two vectors built with the real
  coefficients Im g_n (their Eqs. for the Kubo formula and for the vectors phi_+-).
- Joao and Lopes, J. Phys.: Condens. Matter 32, 125901, arXiv:1810.03732: the first- and
  second-order optical conductivity of any tight-binding model from the Keldysh formalism,
  with the delta and Green's functions expanded in Chebyshev polynomials and the traces
  collected in moments Gamma with one, two and three indices.
- Joao, Viana Parente Lopes and Ferreira, J. Phys. Mater. 5, 045002, arXiv:2110.00574: the
  moments between the Bloch states of the clean cell on a disordered supercell, and the
  disorder self-energy from them.
- de Castro, Ferreira and Bahamon, arXiv:2210.11227: the two-terminal conductance with
  complex absorbing potentials in the leads.
- de Castro, Lopes, Ferreira and Bahamon, PRL 132, 076302, arXiv:2307.09690: the
  Fourier-Chebyshev contraction of the double moments.
- Vidarte, Veiga, Viana Parente Lopes, Cardias, Ferreira, Cysne and Rappoport,
  arXiv:2512.01575: the orbital magnetization of a flake from one Chebyshev expansion.

The statements about pyqula come from reading its code, and the ones about the orbital
field from running it.

The short answer is that most of the linear response of these papers was already in
pyqula: the Kubo-Bastin and the absorptive conductivity of `kpm_optical_conductivity.md`.
Three items were built the same day, the reactive part of the optical conductivity, the
single-shot conductivity and the orbital magnetization (see the last section). What
remains worth taking is a periodic orbital field (the present one is wrong on a 2D
supercell, see the section on the orbital field), the Fourier-Chebyshev contraction of the
DC moments, random twists for the KPM conductivity, an exact Lorentzian line for the KPM
DOS, the disorder self-energy, and the items already recorded as open elsewhere.

## What the papers compute and where pyqula stands

- The density of states and the LDOS from a stochastic trace. pyqula:
  `h.get_dos(mode="KPM")`, with random k-points, and `mode="KPM"` of `h.get_ldos()`, the
  map of every site on the light cone (`kpmtk/ldosmap.py`). Covered.
- The spectral function at a k-point from the Bloch states of the clean cell as start
  vectors (arXiv:2110.00574, the diagonal of its moments). pyqula: the KPM unfolding
  (`unfolding.md`). Covered.
- The Kubo-Bastin DC tensor (arXiv:1410.8140). pyqula: `h.get_dc_conductivity()`.
  Covered.
- The first-order optical conductivity (arXiv:1810.03732). pyqula: the absorptive part
  since 6 November 2026, and the full complex tensor since 14 November 2026,
  `integration="kpm"` of `h.get_optical_conductivity()`. Covered.
- The second-order optical conductivity (arXiv:1810.03732). pyqula: band routes only
  (`conductivitytk/shg.py`, `conductivitytk/photocurrent.py`). Open, recorded in
  `kpm_optical_conductivity.md`.
- The single-shot Kubo-Greenwood conductivity (arXiv:1507.00488). pyqula:
  `mode="singleshot"` of `h.get_dc_conductivity()`, built on 14 November 2026.
- The orbital magnetization of a flake (arXiv:2512.01575). pyqula:
  `h.get_orbital_magnetization()`, built on 14 November 2026.
- The disorder self-energy Sigma(k,w) (arXiv:2110.00574). pyqula: none. New.
- The conductance of a device with complex absorbing potentials (arXiv:2210.11227).
  pyqula: transport with exact lead self-energies only. New, after the single-shot route.
- A uniform out-of-plane field on a periodic supercell, needed for Landau levels and
  the Streda formula in a large disordered cell. pyqula: `h.add_peierls()`, wrong on a
  periodic 2D supercell. New, and a defect.

## The orbital field on a periodic 2D supercell, a defect found on the way

The question was whether pyqula can put a uniform field on a periodic supercell. We test
it with two counts that hold at low flux. Graphene on a torus with $N_\phi$ flux quanta
has $2N_\phi$ states at zero energy, the $n=0$ Landau level, and the square lattice has
$N_\phi$ states in its lowest Landau level. We build the supercell, call
`h.add_peierls(nphi/area)`, take the Bloch Hamiltonian at a k-point, and look at its
Hermiticity, $\max|H-H^\dagger|$, and at its spectrum:

| cell | $N_\phi$ | $\max\lvert H-H^\dagger\rvert$ | levels found | expected |
| --- | --- | --- | --- | --- |
| honeycomb, primitive vectors, 12x12 | 1, 2, 3 | 0.09, 0.17, 0.26 | 0 zero modes | 2, 4, 6 |
| honeycomb, primitive vectors, 12x12 | 12, 24 | 1.0, 1.7 | 2 and 0 zero modes | 24, 48 |
| honeycomb, primitive vectors, 12x12 | 36, 72, 144 | 2.0, 1e-15, 2e-15 | not counted (flux 0.25 to 1 per hexagon) | |
| honeycomb, rectangular cell, 2x60 | 6, 10 | 2.0, 1.0 | 10 and 18 zero modes | 12, 20 |
| honeycomb, rectangular cell, 2x60 | 12 | 4e-15 | 24 zero modes, next level at 0.69t | 24 |
| square, 24x24 | 2, 3 | 0 | no degenerate lowest level | 2, 3 |
| square, 24x24 | 24 | 0 | 24 states at -3.747t, next at -3.257t | 24 |

The zero modes are counted below $10^{-3}t$, except in the second row ($10^{-6}t$). In the
first row the spectrum is that of `eigvalsh` on $H$ itself, which reads one triangle, and
in the second and fourth rows that of $(H+H^\dagger)/2$; a non-Hermitian $H$ has no
physical spectrum, so these counts only say that the answer is wrong.

The pattern is the following. The phase on a bond is $2\pi B\,y\,\Delta x$, the Landau
gauge in Cartesian coordinates, evaluated on every replica. A bond that crosses the
boundary along $\mathbf a_2$ then carries a phase that differs from its translated copy by
$2\pi B\,(\mathbf a_2)_y\,\Delta x$, so the gauge is periodic only if
$B\,(\mathbf a_2)_y\,\Delta x$ is an integer for every such bond, and the same along
$\mathbf a_1$ when $\mathbf a_1$ has a $y$ component. For the honeycomb lattice, whose bonds
have $\Delta x$ of $\pm1/2$ and $\pm1$, this means $N_\phi$ a multiple of $6N$ on the
primitive $N\times N$ supercell and of $6L_1$ on the rectangular $L_1\times L_2$ one, and the
table follows it exactly: 72 and 144 are Hermitian and 36 is not on the first, 12 works and
6 and 10 do not on the second. When the condition fails the Bloch Hamiltonian is not
Hermitian (the honeycomb, whose crossing bonds have $\Delta x\neq0$), or it is Hermitian
with the wrong flux through the plaquettes at the boundary (the square lattice, whose
crossing bonds are vertical). Nothing warns the user. The reference page of the guide
states a weaker condition ("the flux through the unit cell must be an integer number of
flux quanta"), which the 2x60 cell at $N_\phi=6$ and 10 satisfies while failing.

There is a second, independent problem. `add_phase` in `peierls.add_bfield` ends with
`out.todense()`, so a Hamiltonian built with `is_sparse=True` comes back with a dense
`numpy.matrix` (measured on a 20x20 honeycomb supercell). At $10^5$ orbitals that is
160 GB, meaning that no KPM calculation in a field is possible today at the sizes where the
KPM is the method of choice.

The guide and almost every test and example put the field on a ribbon or an island, where
the phase is right. One test does not: `tests/superconductivity/test_nambu_combinations.py`
adds `add_peierls(0.02)` to a periodic 3x3 honeycomb supercell and checks that the field
commutes with `turn_nambu()`, so it compares two non-Hermitian Bloch matrices. The test
still checks what it says, the electron-hole bookkeeping, but on a model that is not
physical. The KPM example `examples/1d/LL_KPM/main.py` is a ribbon, so its phase is right,
but it builds the ribbon with `is_sparse=True` and the field makes it dense.

Two constructions are periodic. The simplest is a Landau gauge on the integer cell
coordinates, a phase $2\pi n\,\Delta R_1 R_2/L_2$ for a hop of $\Delta R_1$ cells along
$\mathbf a_1$ from the cell $R_2$ along $\mathbf a_2$. It is periodic for any integer $n$, a
flux of $n/L_2$ quanta per cell and $nL_1$ in total, but the bonds inside a cell carry no
phase, so the flux through each unit cell is right while its distribution inside the cell
is not that of a uniform field. For graphene, one hexagon per cell, every hexagon gets the
same flux; for a large unit cell, such as a moiré cell, it does not. The construction to
build is the Landau gauge in the fractional coordinates of the supercell,
$\phi=2\pi N_\phi\,s_2\,\Delta s_1$ with the positions of the orbitals, plus the gauge
transformation of the magnetic translation on the bonds that cross the $\mathbf a_2$
boundary. It is uniform inside the cell, periodic for any integer $N_\phi$, and works for
any pair of lattice vectors. It is not derived here: the signs of the boundary term are to
be derived and held to the two counts above at several k-points, to the Hofstadter
spectrum of the square lattice at $N_\phi/L^2=1/q$, and to the Chern number of each
Hofstadter gap through `h.get_chern()`. It must keep a sparse Hamiltonian sparse. Nothing
was repaired.

This is the precondition for every calculation in a field on a periodic supercell, KPM or
not: Landau levels in a disordered supercell of $10^5$ orbitals, the quantum Hall plateaus
from `h.get_dc_conductivity()`, and the Streda formula, $C=(h/e)\,\partial N/\partial B$
from KPM state counts at two commensurate fluxes, which `sparse_kpm_mean_field.md` already
names.

The two scripts behind the last four rows of the table, run with `src` on the Python path:

```python
# honeycomb, rectangular cell, 2x60
import numpy as np
from pyqula.geometrytk import lattices
L1,L2 = 2,60
gs = lattices.honeycomb_lattice_square_cell().get_supercell([L1,L2])
area = abs(gs.a1[0]*gs.a2[1]-gs.a1[1]*gs.a2[0])
for nphi in [6,10,12]:
    h = gs.get_hamiltonian(has_spin=False); h.add_peierls(nphi/area)
    m = np.array(h.get_hk_gen()([0.,0.,0.]))
    herm = np.max(np.abs(m-m.conj().T))
    e = np.sort(np.abs(np.linalg.eigvalsh((m+m.conj().T)/2)))
    print(nphi, herm, np.sum(e<1e-3), "expected", 2*nphi)
```

```python
# square, 24x24
import numpy as np
from pyqula import geometry
L = 24
gq = geometry.square_lattice().get_supercell(L)
for nphi in [2,3,L]:
    h = gq.get_hamiltonian(has_spin=False); h.add_peierls(nphi/L**2)
    e = np.linalg.eigvalsh(np.array(h.get_hk_gen()([0.,0.,0.])))
    print(nphi, -4+2*np.pi*nphi/L**2, np.round(e[:nphi+3],3))
```

## Worth taking, open

### A periodic orbital field

The construction of the previous section. It fixes a silent error and it opens the field
to the KPM, so it comes first.

### The Fourier-Chebyshev contraction of the DC moments

This is the method of arXiv:2307.09690. On the Chebyshev nodes
$\varepsilon_k=\cos(\pi(k+1/2)/M)$ the two broadened deltas become energy-space vectors
$|\phi_k\rangle=\sum_m\frac{2g_m}{\pi}\frac{\cos(m\pi(k+1/2)/M)}{1+\delta_{m0}}|a_m\rangle$,
which is a discrete cosine transform along the moment index of every row of the stored
set. The Kubo-Greenwood kernel at $\varepsilon_k$ is then
$\langle\phi^L_k|\phi^R_k\rangle/(1-\varepsilon_k^2)$, up to the prefactor of their Eq. (7),
and the Kubo-Bastin Hall term needs six such vectors instead of two (their supplement).
This replaces the $NM^2$ of the moment matrix by $N$ transforms of length $M$, meaning
$O(NM\log M)$. The paper measures 50 times at $M=5000$ and 232 times at $M=25000$ against
the moment route, on graphene.

In pyqula the left and right sets $L_n$ and $R_m$ are already built in
`conductivitytk/kpmkubo.py`, so when the whole set fits in memory this is a second
contraction of what exists. When it does not fit, the two schemes differ. pyqula's
`memory=` stores blocks of polynomials, all rows of $B$ of them, and recomputes the left
set $M/B$ times. A cosine transform along the moment index needs all $M$ values of a row at
once, so the paper partitions by rows instead: $S$ slabs of $N/S$ rows, with the whole
recursion repeated for each slab. The $10^6$-orbital run of `kpm_optical_conductivity.md`
held 62 polynomials of 172 to 342, so at that size this needs the new slab loop. Two more
cautions. Our moment product is a zgemm at BLAS speed and our measured runs used
$M=2n_{\rm pol}$ of 170 to 680, so the gain at those $M$ has to be measured; it matters at
small `delta`, where `kpm_optical_conductivity.md` records the $n_{\rm pol}^2$ cost as the
limit. And it applies to the DC routes, where both deltas sit at the same energy, not
directly to the absorptive part at finite $\omega$, which pairs $E$ with $E+\omega$. The
oracle is the moment route on the same random vectors, to the difference of the energy
quadrature.

### Random twists for the KPM conductivity

`kpmkubo.moments()` runs one random vector per k-point of a uniform `klist.kmesh(nk)` in
every sample. Drawing a new twist of the boundary conditions, a random k-point, for each
random vector samples the trace and the k-integral together, and the error bar then
contains the k-sampling. `kpm_optical_conductivity.md` measured the 288-orbital cell at the
Gamma point 40 to 90% away from its own nk=6 limit, which is the finite-size error this
removes at the cost of vectors already paid. The k-point is already an argument of
`ops.hamiltonian(k)`, and the extra cost is one sparse Hamiltonian and its velocities per
sample. The oracle is the band path at a converged mesh, with the deviation inside the
error bar.

### An exact Lorentzian for the KPM DOS and LDOS

pyqula's KPM DOS and LDOS use a kernel (`kpmtk/momenttoprofile.py`), so a KPM curve is a
Gaussian, or roughly a Lorentzian with the Lorentz kernel, and does not lie on the
Lorentzian of the band modes at the same `delta`. The exact series of the resolvent at
complex energy (arXiv:1507.00488, the expansion of G(E + i eta)) gives the Lorentzian
itself, with no kernel, and pyqula already has it in `bsetk/chebyshev.py`. Small.

### The disorder self-energy as a function of k and frequency

This is the method of arXiv:2110.00574. On a disordered supercell they take the Chebyshev
moments between the Bloch states of the clean cell,
$\mu^n_{\alpha\beta}(\mathbf k)=\langle\mathbf k,\alpha|T_n(H)|\mathbf k,\beta\rangle$, with
one recursion per orbital $\beta$, rebuild the $n_0\times n_0$ Green's function
$G(\mathbf k,z)=\sum_nc_n(z)\mu^n(\mathbf k)$ with exact resolvent coefficients, and take
$\Sigma_{\rm dis}(\mathbf k,\omega)=G_0^{-1}-G^{-1}$. This is non-perturbative, and it shows
the k dependence and the concentration dependence that the T-matrix and the
self-consistent T-matrix miss (vacancies in graphene, the Gade singularity). Their cost
estimate is 1 GB and one core-hour for $10^6$ sites at one k-point with 2000 moments.

pyqula's KPM unfolding (`unfolding.md`) already runs those $n_0$ recursions from the
primitive Bloch states, but it keeps only the diagonal $\langle u_\alpha|T_n|u_\alpha\rangle$
for the spectral function. The self-energy needs the full $n_0\times n_0$ overlaps from the
same vectors, the exact resolvent of `bsetk/chebyshev.py`, and two small inversions per
frequency. The oracles are inside pyqula: the single-impurity T-matrix of
`embedding.Embedding` times the concentration, in the dilute limit, and the second-order
Born self-energy (the variance of the onsite disorder times the local Green's function)
for weak Anderson disorder.

### The two-terminal conductance with complex absorbing potentials

This is the method of arXiv:2210.11227. The conductance of a device is the single-shot
trace $\mathrm{Tr}[v_x\,\mathrm{Im}\mathcal G\,v_x\,\mathrm{Im}\mathcal G]$, with the leads
replaced by complex absorbing potentials $-i\Gamma$, so that $H-i\Gamma$ is not Hermitian and
the Chebyshev recursion acquires a damping ($\gamma=\mathrm{asinh}(\ldots)$, their last
equation). They reach twisted bilayer graphene devices of $2.3\times10^6$ sites. pyqula
computes transport with exact lead self-energies (`heterostructures`, `transporttk/`),
which is the oracle on a device of moderate size; this route would only serve devices
beyond what those reach, and it sits on top of the single-shot routine, so it comes after
it.

### Recorded elsewhere

- The second-order conductivity is open in `kpm_optical_conductivity.md`. The formula of
  arXiv:1810.03732 needs moments with one index, $\mathrm{Tr}[h^{abc}T_n]$, two indices and
  three indices, $\mathrm{Tr}[h^aT_nh^bT_mh^cT_p]$, with the generalized velocities
  $h^{a\ldots}$ as derivatives of $H$, and two frequencies: $\omega_1=\omega_2$ gives the
  second harmonic and $\omega_2=-\omega_1$ the photogalvanic response. The three-index set
  is the $NM^3$ that record already names, so this stays the largest item.
- The Chebyshev time evolution is unbuilt in `sparse_kpm_mean_field.md`. Besides the
  propagator, the observables worth taking on a wave packet are its spin, its mean
  position and its variance per direction: the spin gives spin relaxation in a model with
  spin-orbit coupling and disorder, and the growth of the variance gives a diffusion
  constant, a second route to the conductivity and an oracle for the single-shot one
  through the Einstein relation.

## Not taken

A matrix-free Hamiltonian with domain decomposition, which generates the hoppings on the
fly from the unit cell and a list of defects and never stores a sparse matrix, was
considered and not taken. It limits the model to hoppings of a few cells, and for pyqula it
would replace the CSR throughout, while the Kubo route already reaches $10^6$ orbitals in
5.4 GB and the light-cone truncation makes the local quantities linear in the size.

## Built on 14 November 2026

Three items were built the same day: the reactive part of the optical conductivity and the
single-shot Kubo-Greenwood conductivity, both recorded with their measurements in
`kpm_optical_conductivity.md`, and the orbital magnetization, recorded here. The reactive
part was built as the Kramers-Kronig transform of the absorptive part, not with the
first-order formula of arXiv:1810.03732; `kpm_optical_conductivity.md` says why. Building
them also exposed a defect of the Kubo-Bastin route with the Lorentz kernel, recorded there
and left open.

`h.get_orbital_magnetization()` (`topologytk/orbitalmagnetization.py`) computes
M_z = Tr[f(H) m_z]/V on a finite flake with m_z = -(i/2)(xHy - yHx), the commutator form
used by arXiv:2512.01575, electrons of charge -1, positions from the centroid, by
`mode="kpm"` (one-index moments contracted with the Jackson coefficients of the Fermi
function, `delta` setting the number of polynomials as in `get_dos`) or `mode="ED"`. What
was measured:

- The KPM with the exact trace against ED, a 96-site Haldane flake with disorder of width
  0.5, T=0.1, delta=0.01: 2e-4 relative. A translation of the flake changes nothing (1e-15),
  t2 -> -t2 flips every value and a real Hamiltonian gives zero, exactly.
- The Streda slope 2 pi (dM/dE_F)/(C A) inside the gap, with C=1 from `topology.chern` and A
  the number of sites times 3 sqrt(3)/4: 0.816, 0.869, 0.909 and 0.934 at 96, 266, 600 and
  1176 sites, a deficit of about 0.8/n for a hexagon of side n, the strip along the edge
  that the edge current does not enclose. The sign is the one derived from the Maxwell
  relation with pyqula's Hall convention.
- The stochastic trace does not self-average. The relative spread of one random vector on M
  at E_F = -1 is 1.46, 1.13 and 1.57 at 266, 1176 and 4704 sites, because m_z grows with the
  distance to the centroid, so percent accuracy needs about 10^4 vectors at any size, as
  the 800 vectors of arXiv:2512.01575 suggest. The slope in the gap is less noisy, 0.33 per
  vector in units of C at 1176 sites, since the two Fermi levels share their vectors. For a
  few thousand sites the exact trace is the cheaper route. A variance reduction (another
  estimator of the trace, or a local form of the moment) is the open item.
