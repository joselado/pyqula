# Abrikosov pseudofermion mean field

## Abrikosov-pseudofermion (spinon) mean field for Heisenberg models

Let us now address a pure spin model, a spin-$\tfrac12$ Heisenberg model
$H=J\sum_{\langle ij\rangle}\vec S_i\cdot\vec S_j$ in which no electron hops at all, which
`SpinonHamiltonian` treats with the same mean-field machinery. The idea is to write each spin
in terms of an auxiliary fermion, $\vec S_i=\tfrac12 f^\dagger_i\vec\sigma f_i$, the spinon,
subject to the hard local constraint $f^\dagger_i f_i=1$, exactly one fermion per site, since a
spin one half has no charge fluctuations; you can think of a spinon as the spin sector of an
electron, with the charge frozen. The exchange term is then quartic in the spinons, and its
mean-field decoupling produces a bond order parameter $\chi_{ij}=\langle f^\dagger_i f_j\rangle$,
the resonating-valence-bond ansatz, that gives the spinons a dispersion: this is the same
decoupling that the $J$ channel of `h.get_combined_mean_field_hamiltonian()` performs, on a
Hamiltonian with zero bare hopping (a pure spin model has no kinetic term) and with the local
constraint enforced at every site individually, not only on lattice average. The
representation follows the review of Savary and Balents (arXiv:1601.03742, section 4), and
`SpinonHamiltonian` (`pyqula.spinon`) packages exactly this:

```python
from pyqula import geometry
from pyqula.spinon import SpinonHamiltonian

g = geometry.triangular_lattice() # a canonical frustrated-Heisenberg lattice
h = SpinonHamiltonian(g) # zero bare hopping, couplings come from J1/J2/...
h2 = h.get_mean_field_hamiltonian(J1=1.0,nk=12,mix=0.1,maxerror=1e-4)

h2.local_occupation   # <n_i> per site, exactly 1.0 at convergence
h2.constraint_lambda  # converged per-site Lagrange multiplier (local chemical potential)
h2.get_bands()        # spinon dispersion
```

The converged object is a spinon Hamiltonian: its local occupation is one on every site, the
per-site Lagrange multiplier is the local chemical potential that enforces it, and its band
structure is the spinon dispersion, the spectrum of spin-$\tfrac12$ excitations that a quantum
spin liquid would show in place of the magnons of an ordered magnet. `filling=` cannot be
passed here: the representation is only valid at exactly one fermion per site, so that
occupation is imposed site by site through the Lagrange multiplier rather than as a
lattice-averaged Fermi level. Only the U(1) ansatz, with the bond order alone, is
implemented; a Z2 ansatz, which would allow the pairing channel that $J$ can also induce, is
not.

On a frustrated lattice (triangular, kagome and the like) the converged state is
ansatz-dependent, not unique: several distinct self-consistent flux sectors of the bond order
can coexist at the same $J$, and which one an unseeded random `mf` guess lands on is itself
part of the physics rather than noise of the loop, since there is no way to search all
self-consistent solutions and calculations are carried out by assuming a decoupling scheme,
as the review puts it. A chain with one site per unit cell has a unique solution (no
frustration), so repeated calls agree to within `maxerror`; on a frustrated lattice, pass an
explicit `mf=` to select a definite ansatz deliberately rather than comparing energies across
differently seeded runs.

An external Zeeman field couples to $\vec S_i=\tfrac12 f_i^\dagger\vec\sigma f_i$ exactly,
not through any mean-field decoupling, since the term is already bilinear in $f$, so it is
added as an ordinary single-particle term with the same `h.add_zeeman()` or
`h.add_exchange()` used everywhere else in this guide, called on the `SpinonHamiltonian`
instance before `get_mean_field_hamiltonian`:

```python
h = SpinonHamiltonian(g)
h.add_zeeman([0., 0., 0.3])            # or h.add_exchange([0.,0.,0.3])
h2 = h.get_mean_field_hamiltonian(J1=1.0, nk=12)
h2.get_magnetization()                 # induced <S> per site
```

The argument $\vec b$ of `add_zeeman` is the coefficient of $\vec\sigma$ (the Pauli matrices),
not of $\vec S=\vec\sigma/2$, and it enters with a plus sign, as
$+\vec b\cdot\vec\sigma=+2\vec b\cdot\vec S_i$, so the physical field in $H=-\vec h\cdot\vec S_i$ is $\vec h=-2\vec b$,
meaning that the induced $\langle\vec S_i\rangle$ points against the vector passed in; this is
the same convention `add_exchange` uses on an ordinary electronic Hamiltonian elsewhere in
this guide. The one-fermion-per-site constraint is a constraint on the total occupation,
not on the spin, so it stays exactly satisfied under a field while $\langle S_i\rangle$ is
free to grow with it, saturating once the field dominates $J$.

See `examples/spinon/heisenberg_chain/main.py` and
`examples/spinon/heisenberg_chain_zeeman_field/main.py` for runnable versions, and the
notebook `13_spinon_rvb.ipynb` in the folder above.


## Abrikosov-pseudofermion (Read-Newns) mean field for the Kondo lattice

Let us finally address the Kondo lattice, localized moments exchange-coupled to a conduction
electron at the same site, the minimal model of heavy-fermion compounds, which
`KondoLatticeHamiltonian` solves in the same pseudofermion language. Following Coleman's
review (arXiv:cond-mat/0612006, section II.C), the model is written in its
Coqblin-Schrieffer form $H=\sum_k\epsilon_k c^\dagger_kc_k + \tfrac{J}{N}\sum_j
S_{ab}(j)c^\dagger_{jb}c_{ja}$, with $N=2$ for a spin-$\tfrac12$ moment, which is not the
coefficient of a bare $J\vec S_j\cdot\vec s_j$ Heisenberg-form Kondo term (see the caveat
below). Each moment is represented by an Abrikosov pseudofermion
$\vec S_j=\tfrac12 f^\dagger_j\vec\sigma f_j$ subject to the constraint $f^\dagger_jf_j=1$,
as in the previous section, and the exchange term is decoupled into a self-consistent
hybridization field $V_j=-\tfrac{J}{2}\langle f^\dagger_jc_j\rangle$, plus a Lagrange
multiplier $\lambda_j$ enforcing the local constraint. You can think of $V_j$ as the amplitude
for a conduction electron to turn into a spin flip of the local moment and back, a composite
fermion that is half electron and half spin flip, and the resulting mean field is the
large-$N$ (here $N=2$) Read-Newns saddle point of the Kondo-lattice path integral.
`KondoLatticeHamiltonian` (`pyqula.kondolattice`) packages this: given a conduction-electron
Hamiltonian, it fuses on a second, initially decoupled sublattice of localized f-sites (one
per conduction site) with zero bare hopping, and self-consistently solves for $V_j$ and
$\lambda_j$:

```python
from pyqula import geometry
from pyqula.kondolattice import KondoLatticeHamiltonian

gc = geometry.chain()
hc = gc.get_hamiltonian(has_spin=True) # conduction electrons
h = KondoLatticeHamiltonian(hc)

seed = ([0.3+0.0j],[0.0]) # (V,lam), see the caveat below for why
h2 = h.get_mean_field_hamiltonian(J=1.5,filling=0.15,nk=200,mf=seed)

h2.local_occupation   # <n_f> per localized site, exactly 1.0 at convergence
h2.hybridization      # converged V per localized site
h2.constraint_lambda  # converged per-site Lagrange multiplier
```

What comes out is the heavy-fermion band structure: the conduction band hybridizes with the
flat f-level at the energy set by $\lambda_j$, and the converged $V_j$ opens a hybridization
gap around it, with bands on either side that are heavy because most of their weight sits on
the f-sites. The three attributes hold the converged occupation of the f-sites, one per
site, the hybridization and the Lagrange multiplier.

`J` is Coleman's Coqblin-Schrieffer coupling, entering the interaction as $J/N$ with $N=2$,
not the coefficient of a bare $J\vec S_j\cdot\vec s_j$ Heisenberg-form Kondo term; the two
differ by a numerical factor that the review already fixes, and this class follows its
convention exactly. `filling` sets a lattice-wide chemical potential once, from the bare
($V=0$) bands, and holds it fixed: this is a grand-canonical Hamiltonian, in which the
electron count is meant to float as $V$ and $\lambda$ converge. The local
$\langle n_f\rangle=1$ constraint is enforced separately by $\lambda_j$, not by `filling`.

Note that $V=0$ is always itself a self-consistent solution, exactly like the trivial root of
the BCS gap equation, so an unseeded run (`mf=None`, the default) starts there and stays there
even for a `J` that also supports a genuine hybridized state, and a nonzero seed (as above) is
generally needed to find it. Where both solutions coexist, the hybridized state is the true
(lower-energy) ground state. Avoid a `filling` that lands the chemical potential inside the
flat, macroscopically degenerate f-band of the bare problem: at $V=0$ every f-orbital sits at
exactly $\lambda$, so a wide range of fillings, roughly 0.25 to 0.75 for a single conduction
orbital per site, all give the same numerically ill-posed starting point, and `filling=0.15`
above keeps $\mu$ inside the dispersing conduction band instead. The finite Fermi-Dirac
smearing `T` this loop runs at turns the continuous onset $T_K=D\,e^{-1/(J\rho)}$ of the
zero-temperature theory into a genuine finite-temperature Kondo crossover: below a
$T$-dependent threshold in $J$, thermal smearing washes out the hybridization entirely and
$V=0$ becomes the only self-consistent solution, and right at the threshold $V$ jumps directly
to a value of order one rather than growing continuously from zero.

An external Zeeman field couples exactly to both fermion species here, the conduction
electron and the localized moment $\vec S_j=\tfrac12 f_j^\dagger\vec\sigma f_j$, already
bilinear in $f$, so, exactly as for `SpinonHamiltonian` above, it is added as an ordinary
single-particle term with `add_zeeman` or `add_exchange`, called on the
`KondoLatticeHamiltonian` instance before `get_mean_field_hamiltonian`:

```python
h = KondoLatticeHamiltonian(hc)
h.add_zeeman([0., 0., 0.05])
h2 = h.get_mean_field_hamiltonian(J=1.5, filling=0.15, nk=150, mf=seed)
```

`add_zeeman` applies to every site of the fused geometry, that is, to both the conduction and
the f sublattice (offset in $z$); pass a position-dependent callable instead of a constant
vector to target only one of them. The $\langle n_f\rangle=1$ constraint (a constraint on the
total occupation, not on the spin) stays exact under a field. A field competes with the Kondo
singlet: the self-consistent $|V|$ shrinks as the field grows at fixed $J$, and a strong
enough field destroys the hybridized state, which is genuine physics rather than a numerical
failure, and the loop reports non-convergence (`None`) there rather than a spuriously small
but nonzero $V$, exactly the decay toward the always-self-consistent $V=0$ branch that a
subcritical $J$ already produces above.

See `examples/kondolattice/chain/main.py` for a runnable version, and the notebook
`14_kondo_lattice.ipynb` in the folder above.
