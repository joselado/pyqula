# Excitons

## Excitons and the Bethe-Salpeter equation

Let us now go beyond a response at fixed frequency and solve the
two-particle problem itself, with `h.get_bse()` and
`h.get_exciton_energies()`. A mean-field band structure can only ever absorb
above its gap, but the electron and the hole that absorbing a photon creates
attract each other, the pair binds, and the bound state, the exciton, appears
below the gap by its binding energy: the peak below the absorption edge of a
semiconductor. The Bethe-Salpeter equation (BSE) diagonalizes the
electron-hole pair Hamiltonian directly, which gives the exciton energies and
the electron-hole amplitudes that say what each exciton is made of.

The exciton at center-of-mass momentum $Q$ is written as a superposition of the mean field's own transitions,

$$|X\rangle_Q = \sum_{v,c,k} A_{vc}(k)\, c^\dagger_{c,k+Q} c_{v,k} |MF\rangle$$

where $v$ runs over the valence bands, $c$ over the conduction bands and $k$
over the mesh, and the BSE is the eigenvalue problem for the amplitudes
$A_{vc}(k)$, with a kernel made of a direct term (the screened electron-hole
attraction, which binds) and an exchange term (which splits singlet from
triplet and, on its own, reproduces the RPA). The formalism is the
localized-orbital ("point-like orbitals") BSE of the Xatu code,
[arXiv:2307.01572](https://arxiv.org/abs/2307.01572), solved here in its full
non-Tamm-Dancoff form.

By default the interaction is read straight off a converged mean-field
Hamiltonian (`h.V`), so the same interaction that generated the Fock
self-energy inside `h` also generates the BSE kernel and nothing is double
counted. This is time-dependent Hartree-Fock on top of Hartree-Fock: let us
converge a gapped antiferromagnet and ask for its lowest excitons

```python
from pyqula import geometry
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=True)
h.add_sublattice_imbalance(0.6)
hmf = h.get_mean_field_hamiltonian(U=1.5,filling=0.5,mf="antiferro",nk=6) # converge a mean field
es = hmf.get_exciton_energies(nk=6,n=8) # the eight lowest excitons
```

A purely onsite Hubbard `U` is far too short ranged to bind a Wannier-Mott
exciton in two dimensions, so realistic exciton calculations want a
long-ranged interaction instead. `bsetk.interaction.density_interaction`
builds one from the same `U`/`V1`/`V2`/`V3`/`Vr` parameters `Vinteraction`
uses, and it can be handed to any of the exciton methods through `V=`; with
a Coulomb tail on a gapped honeycomb lattice the lowest exciton detaches
from the band edge, and the binding energy says by how much

```python
import numpy as np
from pyqula.bsetk.interaction import density_interaction
h2 = g.get_hamiltonian()
h2.add_sublattice_imbalance(1.0) # a gapped semiconductor
W = density_interaction(h2,Vr=lambda r1,r2: 0.8/np.sqrt((r1-r2).dot(r1-r2)+0.25)) # Coulomb tail
bse = h2.get_bse(V=W,nk=8)
print(bse.get_energies()[0], bse.get_binding_energies()[0]) # lowest exciton, and how far below the gap
```

- `h.get_exciton_energies` returns the exciton energies, `h.get_exciton_binding_energies` how far below the lowest independent-particle transition each one lies (positive means bound), and `h.get_exciton_states` both the energies and the amplitudes $A_{vc}(k)$
- `h.get_bse` returns the full solved object, whose `pairs` attribute holds the k-mesh, the band window and the `(ik,iv,ic)` label of every pair index, so the amplitudes can be resolved in momentum or by band
- `Q=[qx,qy,qz]` gives the excitons at finite center-of-mass momentum, and `h.get_exciton_bands` scans that over a q-path to give the exciton band structure (below); `nv`/`nc` restrict the calculation to the `nv` highest valence and `nc` lowest conduction bands
- `kernel="full"` (default) uses both kernel terms; `"direct"` is the ladder alone (no singlet/triplet splitting), `"exchange"` is exactly the RPA, and `"none"` collapses the spectrum onto the bare transition energies
- `tda=True` applies the Tamm-Dancoff approximation, diagonalizing only the resonant block: four times smaller and Hermitian, and a good approximation at weak coupling

Since the Hamiltonian is spinful, spin is simply part of the orbital index,
so singlet and triplet excitons come out of a single calculation with no
separate spin channel, and the exchange term is what splits them. On a
spin-rotation-invariant reference the lowest transition starts out four-fold
degenerate, and the full kernel resolves it into a three-fold triplet with
the singlet pushed up above it, which neither kernel term produces on its
own.

One caveat on the default `V=None`: `h.V` is only the $z$ channel of an
exchange interaction, so it misses the $x$ and $y$ channels of an anisotropic
run (`J1x`/`J1y` alongside `J1z`), and after `SxSx`/`SySy` it is zero, the
coupling being recorded in the $x$ or $y$ channel. In either case build the
interaction explicitly and pass it as `V=` instead.

The size of the problem is $N_{pair} = n_v n_c N_k$, and the matrix is dense
and $2N_{pair}$ square, so the k-mesh is the expensive knob: `max_memory`
(default 2 GB) refuses a calculation that would not fit rather than letting
it exhaust memory. A gapped reference state is required. A metallic filling
has no well-defined electron-hole pair basis and is rejected, as are
Nambu/BdG Hamiltonians, whose two-particle structure is different.

See
`jupyter-notebooks/functionalities/interacting_mean_field_hamiltonians/17_excitons_bse.ipynb`
for an executed notebook.

### Large k-meshes

How fine a k-mesh an exciton needs is set by how tightly it is bound. A
strongly bound exciton is spread out over the Brillouin zone and is already
converged on a coarse mesh. A weakly bound Wannier-Mott one has an envelope
$A(k)$ peaked sharply at the band edge, and only a fine mesh resolves that
peak. `solver=` says how to reach one

```python
import numpy as np
from pyqula import geometry
from pyqula.bsetk.interaction import density_interaction
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian()
h.add_sublattice_imbalance(1.0) # a gapped semiconductor
W = density_interaction(h,Vr=lambda r1,r2: 0.8/np.sqrt((r1-r2).dot(r1-r2)+0.25))
h.get_bse(V=W,nk=1024,tda=True,solver="iterative",neig=4)
h.get_bse(V=W,nk=65536,tda=True,solver="qtt",neig=1)
```

- `solver="dense"` (default): gives the whole exciton spectrum, and is the only one that works without the Tamm-Dancoff approximation. Good up to a few thousand electron-hole pairs
- `solver="iterative"`: same energies, much finer meshes, but only the `neig` lowest excitons
- `solver="qtt"`: for the very fine meshes a weakly bound exciton needs. Returns the lowest exciton only

Both `"iterative"` and `"qtt"` need `tda=True`, the Tamm-Dancoff
approximation, which is accurate for the weakly coupled systems where bound
excitons live. `"qtt"` is the specialist: it is at its best on a fine mesh
over a narrow band window, which is exactly a Wannier-Mott exciton, and it
is the clear winner in 1D. In 2D, use it on a primitive cell with `nv`/`nc`
kept small; on a supercell its advantage disappears and `"iterative"` is the
better choice. It also wants `nk` to be a power of two, and cannot take an
RPA-screened interaction directly; pass one with
`ScreenedInteraction.get_dict()`.

See `examples/1d/exciton_qtt/main.py` and `examples/2d/exciton_qtt/main.py`
for runnable comparisons of the three solvers, the second plotting the
exciton envelope $|A(k)|^2$ over the Brillouin zone, and
`examples/2d/excitons_bse/main.py` for the lowest exciton of a gapped
honeycomb lattice detaching from the absorption edge as the Coulomb tail is
turned up. See
`jupyter-notebooks/functionalities/interacting_mean_field_hamiltonians/20_bse_matrix_free_qtt.ipynb`
for an executed notebook comparing the three solvers.

### Exciton band structure

An exciton is a two-particle state, so besides its binding energy it has a
dispersion of its own: the bound electron-hole pair propagates with a
center-of-mass momentum $Q$, and $E_X(Q)$ is the exciton band structure. It
is not the difference of two band energies, since the electron-hole
interaction bends it, so its curvature is the exciton's effective mass, and
a flat exciton band means a strongly bound, spatially compact exciton.
`h.get_exciton_bands` solves one BSE per q-point along a path and returns
the result in the same flat form `get_bands` uses

```python
# h2 and W as in the section above
import numpy as np
from pyqula import geometry
from pyqula.bsetk.interaction import density_interaction
g = geometry.honeycomb_lattice()
h2 = g.get_hamiltonian()
h2.add_sublattice_imbalance(1.0) # a gapped semiconductor
W = density_interaction(h2,Vr=lambda r1,r2: 0.8/np.sqrt((r1-r2).dot(r1-r2)+0.25))
# nv=nc=2 keeps both members of each spin-degenerate pair
opts = dict(V=W,nq=20,nk=8,nv=2,nc=2,n=4)
qs,es = h2.get_exciton_bands(**opts) # the four lowest excitons along the path
qs0,es0 = h2.get_exciton_bands(kernel="none",**opts) # the bare continuum
```

`qs` holds the integer index of the q-point along the path and `es` the
exciton energy, both flat 1D arrays ready for a scatter plot; `n` keeps only
the `n` lowest excitons at each q-point and every other argument is passed
straight to `get_bse`. `qpath` takes the same input as `get_bands`, a list
of high-symmetry labels or of explicit q-vectors, and $Q$ is not restricted
to the k-mesh, since the pair basis diagonalizes at $k$ and $k+Q$
independently. Running it a second time with `kernel="none"` is the same
call with the kernel construction skipped, and gives the bottom of the
electron-hole continuum to plot the exciton band against, which is what
makes the binding visible: the exciton band runs below the continuum all
along the path. If the mean-field reference is unstable against some
excitation at some q-point, `es` comes back complex there rather than
silently losing the imaginary part. The cost is `nq` full dense
diagonalizations, so `nv`/`nc` and `tda=True` are the knobs that make a long
path affordable, and `parallel.set_cores` parallelizes over the path.

One trap is invisible at $Q=0$: `nv`/`nc` must not cut a degenerate
multiplet in half. Every band of a spinful Hamiltonian with no spin-orbit
coupling and no magnetic order is two-fold degenerate. Setting `nv=1` there
keeps an arbitrary state out of that degenerate pair, and the exciton
energies inherit the arbitrariness: on a time-reversal-symmetric model,
$E_X(Q)$ visibly stops being even in $Q$. The library warns when the window
splits a multiplet; use an even `nv`/`nc` on a spin-degenerate Hamiltonian.

See `examples/2d/exciton_bands/main.py` for a runnable version: the exciton
band of a gapped honeycomb lattice, plotted below the electron-hole
continuum of the same model.

### The screened interaction

Everything above uses the bare interaction in both kernel terms, which makes
the BSE time-dependent Hartree-Fock. But an electron and a hole added to a
solid do not feel the bare interaction: the other electrons rearrange around
them, and what survives that rearrangement is the screened interaction.
Since the mean-field step has already left the bands $\{e_n(k), C^n(k)\}$ on
a k-mesh, the static RPA screening can be computed from them directly rather
than postulated,

$$\chi^0_{ab}(q) = \frac{1}{N}\sum_k \sum_{n,m} (f_{nk}-f_{m,k+q})\, \frac{\rho^{nm}_a(k,q)\, \rho^{nm*}_b(k,q)}{e_{nk}-e_{m,k+q}}, \qquad \rho^{nm}_a(k,q) = C^{n,k*}_a C^{m,k+q}_a$$

$$\varepsilon(q) = 1 - v(q)\chi^0(q), \qquad W(q) = \varepsilon^{-1}(q)\, v(q)$$

with $a,b$ running over the spin-orbitals of the unit cell, in the same
point-like-orbital approximation the rest of the BSE uses. `screening="rpa"`
computes this and puts $W$ in the direct (ladder) term

```python
import numpy as np
from pyqula import geometry
from pyqula.bsetk.interaction import density_interaction

h = geometry.honeycomb_lattice().get_hamiltonian()
h.add_sublattice_imbalance(1.0) # a gapped semiconductor

# a BARE interaction: an onsite term plus a soft-cutoff Coulomb tail
coulomb = lambda r1,r2: 0.6/np.sqrt((r1-r2).dot(r1-r2)+0.25)
V = density_interaction(h,U=1.0,Vr=coulomb)

bare = h.get_bse(V=V,nk=8) # time-dependent Hartree-Fock
screened = h.get_bse(V=V,nk=8,screening="rpa") # GW-BSE style
```

What you see when comparing the two is a weaker binding in the screened
calculation, the exciton moving closer to the band edge, since the medium
partly cancels the attraction between the electron and the hole.

The exchange term keeps the bare interaction whatever `screening` is set to.
This is the standard GW-BSE split rather than an oversight, since screening
the exchange term too would count the same bubbles twice. One consequence is
that `kernel="exchange"` is unaffected by `screening`, and a warning is
raised if both are given.

A fitted Hubbard $U$ must not be screened. A $U$ chosen to reproduce a
material is already an effective, screened interaction, and screening it
again double counts and gives a spuriously weak interaction. Use screening
for a genuinely bare interaction: a long-range Coulomb tail from
`density_interaction(Vr=...)`, or bare `V1`/`V2`/`V3` shells. The dangerous
case is the default `V=h.V` taken from a Hubbard calculation. This is a
different question from combining an RPA $W$ with a BSE ladder, which is the
standard GW-BSE construction and is not double counting.

`screening="crpa"` is the constrained variant, which leaves the transitions
inside the `nv`/`nc` BSE band window out of the polarization
([arXiv:0710.4013](https://arxiv.org/abs/0710.4013)). That is the right
choice when the band window is being treated as a downfolded model to be
solved exactly afterwards, and it screens strictly less than the full RPA.
It needs a genuine subset of the bands: with the default `nv=nc=None` the
window is the whole spectrum, nothing is left outside it to do the
screening, and the call is refused rather than silently returning the bare
interaction.

The screened interaction is also available on its own, as a `ScreenedInteraction`

```python
W = h.get_screened_interaction(V=V,nk=8)
print(W.epsmin) # smallest dielectric eigenvalue over the mesh
Wq = W.at(W.qs[3]) # the screened interaction at one q-point
d = W.get_dict() # ... and back in real space, usable at any q
```

$W(q)$ is not short ranged, so unlike the other interactions in this guide
it lives only on the mesh it was computed on, which is exactly the set of
$q$ the kernel needs. Asking for it at any other $q$ raises rather than
silently using the nearest point. `get_dict()` gives the real-space version,
which can be evaluated anywhere, inspected to see how far the screened
interaction reaches, or passed to `get_mean_field_hamiltonian(V=...)` for a
screened-exchange mean field. `nkW` sets a finer mesh for the screening than
for the exciton, and must be an integer multiple of `nk`.

Where the dielectric matrix is built matters. Screening is a property of the
charge channel: what polarizes the medium is the total density, and what the
medium's induced charge acts back on is again the total density. So
`channel="charge"` (the default) builds $\varepsilon$ on site indices,

$$\varepsilon_{ij}(q) = \delta_{ij} - \sum_k v^c_{ik}(q)\,\chi^c_{kj}(q), \qquad \chi^c_{ij} = \sum_{\sigma\sigma'}\chi^0_{(i\sigma)(j\sigma')}, \qquad v^c_{ij} = \tfrac{1}{4}\sum_{\sigma\sigma'} v_{(i\sigma)(j\sigma')}$$

and adds the resulting correction, which is spin-independent, to the bare interaction:

$$W_{(i\sigma)(j\sigma')} = v_{(i\sigma)(j\sigma')} + \left[v^c \chi\, v^c\right]_{ij}, \qquad \chi = \chi^c(1-v^c\chi^c)^{-1}$$

This is the standard GW construction. One convention matters here: off site
$v^c_{ij}=V_{ij}$, but on site $v^c_{ii}=U/2$ rather than $U$, because a
Hubbard term couples only opposite spins, so only half of a site's own
density acts on a given electron.

Building the dielectric matrix in the charge channel keeps spin-rotation
invariance exact: the Ising $S^z_iS^z_j$ part of $W$ is left exactly as the
bare interaction had it. That is why it is the default. The alternative,
`channel="orbital"`, dresses the full spin-orbital matrix as
$W = \varepsilon^{-1}v$. It screens the charge and spin channels with a
single density-density kernel and therefore breaks SU(2): an exciton
multiplet that should stay degenerate splits visibly in the orbital channel
and stays degenerate in the charge one. Use the default. The two channels
coincide exactly for a spinless Hamiltonian.

Screening does not always weaken the interaction. It does when the
charge-channel interaction $v^c(q)$ is positive definite at every $q$, as a
physical repulsion is, since the static charge response is negative and
$\varepsilon=1-v^c\chi^c$ then only reduces it. With no onsite term the
interaction matrix has zero trace, so it has negative eigenvalues, and
screening enhances the interaction along them rather than reducing it, in the
same way an RPA kernel enhances a magnetic instability. A Coulomb tail with
no onsite term is simply missing its largest matrix element, so include a
realistic $U$, and note that $U>0$ is not enough: the eigenvalues of $v^c(q)$
are $U/2$ plus those of the Fourier transform of the intersite interaction,
which averages to zero over the Brillouin zone and so is negative somewhere
(on the square lattice a first-neighbor $V_1$ gives $U/2-4V_1$ at the corner
of the zone), and $U/2$ has to outweigh the most negative of them.

Finally, if an eigenvalue of $\varepsilon(q)$ actually reaches zero, the RPA
has diverged: that is a charge or spin instability of the mean field at that
wavevector, the same $1-V\chi=0$ condition `h.get_rpa_kernel_poles` reports
as a collective mode, and the call raises rather than returning a huge
number.

The parameters, in short:

- `screening` picks the interaction in the direct term: `None` for the bare one, `"rpa"` for the screened $W$, `"crpa"` for the constrained version when the band window will be solved exactly afterwards
- `channel` picks where the dielectric matrix is built. Leave it at `"charge"`, which keeps spin-rotation invariance exact
- `nkW` is the mesh the screening is computed on. Raise it above `nk`, by an integer factor, when the screening needs more k-points than the exciton does

See `examples/2d/screened_bse/main.py` for a runnable version, and
`jupyter-notebooks/functionalities/interacting_mean_field_hamiltonians/21_screened_interaction.ipynb`
for an executed notebook.
