# Spin susceptibility, RPA and magnons

## Spin susceptibility and RPA

The methods above compute the bare response, the one of non-interacting
electrons. We will now see how the response of an interacting system is
obtained at the RPA level, with `h.get_spinchi_ladder()` and
`h.get_spinchi_full()`. The idea is that the perturbation also changes the
mean field the electrons feel, which moves more electrons, and summing that
series gives $\chi_{RPA} = \chi(1-U\chi)^{-1}$: the bare response dressed
with the Hubbard `U` stored on a mean-field Hamiltonian (`h.V`, see
"Interactions at the mean-field level"). For a magnetically ordered state
this is what carries the magnons, which the bare response does not have. Let
us converge an antiferromagnetic chain and ask for its transverse spin
response at a small momentum transfer

```python
from pyqula import geometry
import numpy as np
g = geometry.bichain() # two sites per cell, so Neel order fits in the cell
h = g.get_hamiltonian(has_spin=True)
seed = h.copy() ; seed.add_antiferromagnetism(0.5) # symmetry-breaking seed
hmf = h.get_mean_field_hamiltonian(U=3.0,filling=0.5,mf=seed,nk=100) # magnetic state
(es,chis) = hmf.get_spinchi_ladder(energies=np.linspace(0.,2.,100),q=[0.1,0.,0.],nk=40,delta=2e-2)
```

What you see in the imaginary part of `chis` is a sharp peak at the magnon
energy at that $q$, which is what inelastic neutron scattering measures in a
magnet; without spin-orbit coupling the peak goes to zero energy at $q=0$,
the Goldstone mode of the broken spin-rotation symmetry.

- `get_spinchi_ladder` computes the transverse ($S^+/S^-$) response, the spin-wave-like excitations
- `get_spinchi_full` computes the full $(S_x,S_y,S_z)$ tensor response
- `RPA=True` (default) dresses the response with the interaction; `RPA=False` returns the bare response
- `h.get_qdos_iets` scans `get_spinchi_full` over a q-path instead of a single q, giving the spin-excitation dispersion along high-symmetry directions

With `q` left out, the response is dressed at every $q$ of the mesh and
averaged afterwards, which is the local response, the one a probe sitting on
a single site sees: the RPA series is summed at a fixed momentum transfer, so
each $q$ has to be dressed on its own before the average, and dressing the
averaged bare response instead would not be the response of any state. This
local response is what `h.get_iets_ldos` builds its map from.

The `"G","K","M"` path labels need a 2D lattice; on the antiferromagnetic
honeycomb lattice the dispersion map along the path is

```python
g2 = geometry.honeycomb_lattice() # already two sublattices per cell
h2 = g2.get_hamiltonian(has_spin=True)
hmf2 = h2.get_mean_field_hamiltonian(U=3.0,filling=0.5,mf="antiferro")
qdisp = hmf2.get_qdos_iets(energies=np.linspace(0.,2.,60),qpath=["G","K","M"],nq=30,nk=20,delta=1e-2)
```

The real-space counterpart is `h.get_iets_ldos`, which computes the response
at a single energy resolved in space, a map of the spin-flip signal that
inelastic tunneling spectroscopy (IETS) sees with an STM; combined with
`h.get_ldos` it gives the elastic and inelastic STM maps side by side.

Both snippets depend on the mean-field state actually being ordered, which
is the first thing to check, with `hmf.get_vev("sz")`, before reading
anything into the excitation spectrum: an RPA calculation on top of an
unpolarized reference state runs perfectly happily and tells you nothing
about magnons. Two things decide it. The unit cell must be able to hold the
order: a one-site `geometry.chain()` cell cannot represent Neel order at all,
and seeding it with `mf="antiferro"` converges to exactly zero moment, while
`bichain` and `honeycomb_lattice` both have the two sublattices. And `U` must
be past the ordering transition: on the honeycomb lattice at half filling
`U=2` gives a moment of 0.002, meaning none, while `U=3` gives 0.44. Note
also that the cost of `get_qdos_iets` is the product of its `nq`, `nk` and
energy-grid sizes, so it grows quickly: the grid above takes well under a
minute, while a 80x40x100 one is closer to an hour.

See `examples/1d/rpa/main.py` (RPA spin response vs q for an antiferromagnetic chain), `examples/2d/rpa_triangular/main.py`/`examples/2d/rpa_honeycomb/main.py` (`get_qdos_iets` dispersion along a q-path) and `examples/0d/rpa_island/main.py`/`examples/0d/rpa_finite_chain/main.py` (`get_iets_ldos` real-space IETS maps) for runnable versions, and the executed notebooks
`jupyter-notebooks/functionalities/interacting_mean_field_hamiltonians/09_rpa_beyond_onsite.ipynb`
and `10_rpa_collective_modes.ipynb` for what the spectra look like.

#### Running the response on a GPU

All of these RPA functions, `get_spinchi_ladder`, `get_spinchi_full`,
`get_qdos_iets`, `get_iets_ldos`, `get_rpa_kernel_poles`,
`get_magnon_bands(method="rpa")` and the density-channel
`get_densitychi_RPA`, bottom out in the same Lindhard kernel, a sum over
pairs of eigenstates at every k-point whose cost grows as the fourth power of
the number of sites in the cell. Calling `gpu.set_gpu(True)` first runs that
kernel on a GPU instead, which pays off once the cell holds more than a few
sites (see "Running on a GPU"). The pair basis, which the same functions
take for an interaction that couples different sites, has its own kernel on
the device and follows the same switch, so the route the interaction picks
does not change whether the calculation runs on a GPU.

The precision of that sum is chosen with `chi_prec`. On the GPU it is
`"single"` by default, since the double-precision arithmetic of a consumer
card can be an order of magnitude slower than its single-precision one, and
`chi_prec="double"` switches it back; the CPU takes both values too, with
`"double"` as its default. The eigenstates are obtained in double precision
either way and the response comes back as double-precision numbers, so what
single precision changes is only the rounding of the sum, which leaves the
bare response within about $10^{-7}$ of the double-precision one

```python
from pyqula import geometry
from pyqula import gpu
import numpy as np
gpu.set_gpu(True) # run the GPU kernels of the package on the device
g = geometry.honeycomb_lattice() # two sites per cell
h = g.get_hamiltonian() # first-neighbor hopping
hmf = h.get_mean_field_hamiltonian(U=3.0,filling=0.5,mf="antiferro",nk=10) # Neel mean field
es = np.linspace(0.01,1.0,50) # frequencies
(es,chi) = hmf.get_spinchi_full(q=[0.2,0.,0.],nk=10,energies=es,delta=0.05,
               chi_prec="double") # the response in double precision
```

Note that the RPA dressing amplifies that rounding close to an instability or
a Goldstone mode, where $1-U\chi$ is nearly singular: at $q=0$, next to the
Goldstone mode of the Neel antiferromagnet above, the dressed response
differs from the double-precision one by a few parts in $10^4$, so a
calculation that needs the dressed response at a pole to more digits than
that should use `chi_prec="double"`.

### RPA kernel poles and magnon bands

Let us now find the collective modes directly, with
`h.get_rpa_kernel_poles()` and `h.get_magnon_bands()`. The RPA-dressed
response diverges wherever the kernel $1-U\chi(q,\omega)$ becomes singular,
and those poles are the collective modes of the system, spin waves (magnons)
and plasmons; a kernel eigenvalue crossing zero at $\omega=0$ instead is a
Stoner or RPA instability. `h.get_rpa_kernel_poles` scans a frequency window
at a fixed `q` and returns every such pole, for any pair of operators `A`,`B`
and any interaction matrix `V`, defaulting to the charge channel and $q=0$
like the other generic response functions above

```python
from pyqula import geometry
import numpy as np
g = geometry.bichain() # two sites per cell, so Neel order fits in the cell
h = g.get_hamiltonian()
seed = h.copy() ; seed.add_antiferromagnetism(0.5) # symmetry-breaking seed
hmf = h.get_mean_field_hamiltonian(U=3.0,nk=100,mf=seed,filling=0.5)
N = len(g.r) # the response matrix is one entry per site
V = 3.0*np.identity(N) # charge-channel interaction, in site space
poles = hmf.get_rpa_kernel_poles(V=V,q=[0.1,0.,0.],
        energies=np.linspace(0.,4.,200),delta=2e-2,nk=40)
```

Two things there are easy to get wrong. As above, the mean field needs a unit
cell that can hold the order being sought, which is why `bichain` and not
`chain` is used, and the converged state has `hmf.get_vev("sz")` equal and
opposite on the two sublattices. And `V` must live in the same space as the
response matrix: `get_rpa_kernel_poles` defaults to the charge channel, one
entry per site, so `V` is `N`x`N`. It is not `hmf.V`, which is the mean-field
interaction in spin-orbital space (`2N`x`2N`) and raises a dimension error
here. For the spin channel use `get_magnon_bands` below, which builds the
$S_x,S_y,S_z$ vertex from `hmf.V` itself.

`poles` is an `(npoles,2)` array, one row per collective mode found: the pole
frequency and its residual imaginary part. The latter is signed (it is the
kernel eigenvalue's actual imaginary part at the crossing, which can lie on
either side of the real axis), so judge how sharp a mode is by its
magnitude: a small `abs(gamma)` means a sharp mode, a large one means it is
heavily damped or the crossing is numerical noise.

`h.get_magnon_bands` does this in the full $S_x,S_y,S_z$ spin channel, taking
the interaction from the converged mean field, and scans along a q-path, so
that the result is the magnon dispersion of the magnetically ordered state

```python
qs,ws,gammas = hmf.get_magnon_bands(nq=40,energies=np.linspace(0.01,3.,200),delta=2e-2,nk=40)
```

Different q-points can have different numbers of poles, so `qs`, `ws` and
`gammas` come back as flat arrays of equal length, ready for a scatter plot.
`qs` is the index of the q-point along the path, the same convention
`get_bands` uses; `ws` is the pole frequency and `gammas` its residual
imaginary part, and filtering on `np.abs(gammas) < threshold` keeps only the
sharp branches. What you see for the antiferromagnetic chain is an acoustic
branch going to zero at $q=0$, the Goldstone mode, and an optical one above
it. See `examples/1d/magnon_bands/main.py` for a runnable version.

### Interactions beyond onsite

The `V` passed to `get_rpa_kernel_poles` is not restricted to a single onsite
matrix. It can also be a real-space hopping-like dictionary
`{(n1,n2,n3): matrix}`, keyed by lattice-vector offset in the same
convention as `h.get_hopping_dict()`, for an interaction with support beyond
the same unit cell. It is Fourier-transformed to $V(q)$ at whatever `q` the
response is evaluated at, using the same Bloch-phase convention as the
Hamiltonian's own hoppings, so that an extended interaction is dressed
exactly like an extended hopping. For a nearest-neighbor $V_1$ on a chain
this gives the expected $V(q)=2V_1\cos{2\pi q}$, meaning that the interaction
is repulsive at $q=0$ and attractive at the zone boundary

```python
import numpy as np
from pyqula import geometry
g = geometry.chain()
h = g.get_hamiltonian(has_spin=False)
N = h.intra.shape[0] # one site per cell here
I = np.identity(N)
V = {(0,0,0): 0.0*I, (1,0,0): 0.6*I, (-1,0,0): 0.6*I} # nearest-neighbor V1
poles = h.get_rpa_kernel_poles(V=V,q=[0.5,0.,0.],energies=np.linspace(0.,2.,60),
                                delta=2e-2,nk=200)
```

Note the channel: `get_rpa_kernel_poles` dresses the charge response by
default, so a dictionary `V` here must be a density-density interaction in
site space, of the same dimension as the response matrix
`chiAB(...,mode="matrix")` returns. It is not the place to put a spin vertex.

What happens in the spin channel when the interaction couples two different
sites? A vertex with one index per site carries the Hartree part of such an
interaction, but not its Fock part, which acts on the electron-hole pair index
$(i,j)$ and has nowhere to go in that basis. For a neighbor-shell $V_{ij}$ that
Fock part is all there is in the spin channel, so on a ferromagnetic chain
ordered by $V_1$ alone the site vertex vanishes and no magnon comes out at all;
for an exchange $J_1$ it is a correction of a few percent to the dispersion at
finite $q$. So `get_spinchi_full`, `get_spinchi_ladder`, `get_iets_ldos` and
`get_magnon_bands` use the site vertex only for an interaction that couples
each site to itself, a Hubbard $U$, where it is exact for the transverse
response, and sum the ladder in the pair basis of `method="pair"` (below) for
anything that couples different sites, in the same cell or in different ones. The call is the same either way,
with the same `energies`, `delta`, `nk` and temperature `T` (equal to `delta`
unless given); what the pair basis does not have is the Nambu basis, which
is refused there rather than computed with the site vertex. With `q` left out, the response is averaged over the k-mesh after
dressing each $q$, which is the local RPA response.

### The three magnon routes

`h.get_magnon_bands` takes a `method`, and the three cover different
interactions because they keep different amounts of the ladder:

| interaction | `"rpa"` | `"pair"` | `"tdhf"` |
|---|---|---|---|
| onsite Hubbard $U$ | yes | yes | yes |
| neighbour-shell density-density $V_1,V_2,\dots$ | through the pair basis | yes | yes |
| exchange $J_1,J_2,\dots$ (isotropic or anisotropic) | through the pair basis | yes | yes |
| metallic reference | yes | yes | with `metal=True` |
| non-collinear (canted, spiral) state | yes | yes | yes |
| frequency-resolved $\chi(\omega)$ | yes | yes | no (an eigenproblem) |

The middle row splits because of which index the interaction is diagonal in.
Writing $H_{int} = \tfrac12\sum V_{ij} n_i n_j$, the transverse rung is
$K_{(ij),(kl)} = -V_{ij}\delta_{ik}\delta_{jl}$, diagonal in the pair index
$(i,j)$ rather than in the site index. An onsite $U$ collapses this onto the
pairs with $i=j$, which is exactly what the site vertex of `method="rpa"`
works with, and that is why it is exact there. An extended $V_{ij}$ lives in
the pairs with $i\ne j$, which that basis simply does not contain, and this is
why `method="rpa"` hands such an interaction to the pair basis.

`method="pair"` solves the same ladder in the pair index, where $V$ is
diagonal. Only the pairs the interaction actually connects enter, so a
short-ranged $V$ costs $N(z+1)$ rather than $N^2$. It keeps the frequency
scan, needs no gap, and assumes nothing about the spin structure of the
state. Let us take the honeycomb lattice with both an onsite $U$ and a
first-neighbor $V_1$, converge its antiferromagnetic state, and ask for the
magnons and the transverse response

```python
import numpy as np
from pyqula import geometry
from pyqula.meanfield import VJinteraction
nk = 6                                  # the SCF and the magnon share this mesh
g = geometry.honeycomb_lattice()
hmf = VJinteraction(g.get_hamiltonian(),U=3.0,V1=0.5,filling=0.5,
                     mf="antiferro",nk=nk,maxerror=1e-10).hamiltonian
qs,ws,gammas = hmf.get_magnon_bands(method="pair",nq=20,
                    energies=np.linspace(1e-3,3.,400),delta=1e-3,nk=nk)
es,chi = hmf.get_transverse_spinchi(energies=np.linspace(0.,2.,100),
                    q=[0.1,0.,0.],delta=1e-2,nk=nk)   # (Sx,Sy,Sz) x site chi(w)
```

The kernel has two terms, an exchange one and a Hartree one. For a state
with a single spin quantization axis the Hartree term drops out of the
transverse response, and keeping only the exchange term is exact. For a
non-collinear state the two spin sectors mix, dropping the Hartree term
breaks SU(2), and the acoustic branch comes out spuriously gapped. Both terms
are therefore kept, which keeps the Goldstone mode in place for a canted or
spiral state as well.

The last row of the table is what `"tdhf"` is for: it gives the magnon
energies as eigenvalues, with no frequency grid and no broadening, which is
what the Goldstone residual is measured on. `"pair"` and `"tdhf"` agree with
each other for every interaction in the table, and in a metal both reproduce
the exact saturated-ferromagnet dispersion; `"rpa"` agrees with them for an
onsite $U$, and for anything longer ranged it is `"pair"`.

What about exchange? An exchange interaction $J\,\vec S_i\cdot\vec S_j$ is
not a density-density one: its Ising part $J S^z_iS^z_j$ is, but its
transverse part $J/2(S^+_iS^-_j+\mathrm{h.c.})$ flips a spin on each site,
and no matrix of the form $V_{ij}n_in_j$ can write it. The way the mean field
handles it is by writing $S^x_iS^x_j$ and $S^y_iS^y_j$ as the same Ising term
in two rotated spin frames, and the self-consistent calculation records the
three spin channels on the Hamiltonian it returns. `"pair"` and `"tdhf"` read
them and build the kernel of each channel in its own frame, which is exactly
the derivative of that mean field, so the transverse part is included and
the Goldstone mode sits at zero for an isotropic $J$ as it does for $U$. This
requires a Hamiltonian from `VJinteraction` or from
`get_mean_field_hamiltonian` with `J1`/`J2`/`J3`/`Jr`, which are the ones
that record the channels.

The site vertex would get this only partly right. It keeps the Goldstone mode
for an isotropic $J$, but the exchange bonds also enter through a Fock term on
the pair index, as $V_1$ does: on the antiferromagnetic honeycomb lattice with
$J_1=3$ at `nk=6` it puts the acoustic magnon at $q=0.1$ at 1.3296, against
1.3687 from `"pair"` and `"tdhf"`, which a brute-force reference confirms.
This is why `"rpa"` sums an exchange interaction in the pair basis.

The single-axis couplings `SzSz`, `SxSx` and `SySy` record their channel too,
as an anisotropic exchange with the other two at zero, so all three routes
take them. A global spin rotation of a converged Hamiltonian keeps an
isotropic exchange exact, but it would turn an anisotropic one into an
interaction with $S^x_iS^y_j$-like cross terms that the three channels cannot
hold, so `h.global_spin_rotation()` refuses it, unless the rotation is about
the axis of a uniaxial exchange; rotate the Hamiltonian before the mean-field
calculation instead.

See
`jupyter-notebooks/functionalities/interacting_mean_field_hamiltonians/19_magnons_pair_ladder.ipynb`
for an executed notebook comparing the pair basis with the time-dependent Hartree-Fock magnons.

### Magnons from time-dependent Hartree-Fock

You can think of a magnon as the same kind of object as an exciton: a bound
two-particle excitation of a mean-field state, made of electron-hole pairs
whose electron and hole have opposite spin rather than the same spin. So the
Bethe-Salpeter equation of the exciton sections below already describes it,
and `method="tdhf"` is that equation restricted to the spin-flip pairs.
Because it works in the pair index, it carries a neighbor-shell interaction
properly

```python
import numpy as np
from pyqula import geometry
from pyqula.meanfield import VJinteraction
nk = 6                                  # the SCF and the magnon share this mesh
g = geometry.honeycomb_lattice()
scf = VJinteraction(g.get_hamiltonian(),U=3.0,V1=0.5,filling=0.5,
                     mf="antiferro",nk=nk,maxerror=1e-10)
hmf = scf.hamiltonian
print(hmf.get_goldstone_residual(nk=nk))          # 2e-10: the mode is exact
qs,es = hmf.get_magnon_bands(method="tdhf",nk=nk,nq=20,n=2)
```

`qs`,`es` are flat 1D arrays in the same convention as the RPA
`get_magnon_bands` above (`qs` is the integer index along the q-path), and
`es` is complex whenever the mean-field state is unstable against some
excitation. `h.get_magnon_energies(Q=...)` gives the spectrum at a single
momentum.

The check that this is right is the Goldstone theorem. A state that orders
magnetically without spin-orbit coupling breaks SU(2) spontaneously, so a
uniform spin rotation costs no energy and there must be a magnon at exactly
zero energy at $Q=0$. `h.get_goldstone_residual()` measures how far the
calculation is from that. It comes out proportional to the tolerance the
mean field was converged to and to nothing else, roughly $10^{-10}$ for a
mean field converged to `maxerror=1e-10`, so check it before reading any
dispersion. It is deliberately not "the eigenvalue nearest zero", which
converges much more slowly; a small imaginary part of that size on the
acoustic branch is expected rather than a problem.

Three things matter before using it. The first is that the mean field and
the magnon must use the same `nk`: a mean field converged at `nk=20` and a
magnon solved at `nk=4` is not self-consistent on the magnon's mesh, and the
acoustic branch picks up a large spurious gap. The Hamiltonian does not
record the mesh it was converged on, so `get_goldstone_residual` is how to
catch this.

The second is that a metallic reference needs `metal=True`. By default the
same number of bands is taken as occupied at every k-point, which a metal
does not have; `metal=True` decides the occupied and empty sets separately
at each k-point instead, which is what an itinerant magnet needs, and in
particular a ferromagnet ordered by a neighbour-shell $V_1$ alone. It
changes nothing for a gapped reference, so it is safe to leave on when
unsure. Two things change once it runs. The magnon sits inside the Stoner
continuum and is no longer the lowest mode, so
`h.get_magnon_bands(method="tdhf", metal=True, by="weight")` picks branches by
how much spin character they carry rather than by energy. And $E(q)$ is even
in $q$ only if the occupied set is symmetric under $k\to-k$, which a finite
mesh need not make it: otherwise the $+q$ and $-q$ magnons genuinely differ.
Choose the mesh so the occupied set is symmetric.

The third is where the interaction came from. An isotropic exchange
(`J1`/`J2`/`J3`/`Jr`) is taken with its transverse part
$J/2(S^+_iS^-_j+\mathrm{h.c.})$, through the spin channels the
self-consistent calculation recorded, as described in "The three magnon
routes" above, and it has its Goldstone mode like any other interaction.
Two cases raise `ValueError` instead, each with a message saying why. The
first is a hand-built exchange matrix: it carries only an Ising term and no
record of its spin channels, which is what an isotropic exchange that lost its
transverse part would look like too, and solving that would give an
ordinary-looking dispersion gapped by of order $J$ where zero is required. The
second is an anisotropic exchange (`J1x`, `J1y`, `J1z` unequal, or the single
axis of `SzSz`, `SxSx` and `SySy`), which breaks spin rotation symmetry
explicitly, meaning that its magnon gap is real and there is no Goldstone mode
to check; `check_su2=False` computes that spectrum, with every channel
included, and it is the same for `SzSz`, `SxSx` and `SySy` at the same $J$.

For a collinear state only the spin-flip pairs are needed, which is exact
and much cheaper, and that is the default. A canted or spiral state has no
such block, and the whole pair basis is used instead without your having to
ask (`channel="all"` forces it). This covers genuinely non-collinear states
such as the 120-degree spiral of the triangular-lattice Hubbard model, whose
Goldstone mode survives to the tolerance the mean field was converged to. On
a plain onsite Hubbard `U`, where both this and `method="rpa"` are exact,
the two agree.

See `examples/2d/magnon_bands_tdhf/main.py` for a runnable version, and
`jupyter-notebooks/functionalities/interacting_mean_field_hamiltonians/18_magnons_tdhf.ipynb`
for an executed notebook.

### Density (charge) response

The same dressing can be done in the charge channel, with
`h.get_densitychi_RPA()` and `h.get_plasmon_bands()`, the density-density
analogs of `get_spinchi_full` and `get_magnon_bands`: the collective modes
are now the plasmons, or the charge-order modes, of a `V1`/`V2`/`V3`
neighbor-shell (plus onsite `U`, plus a general `Vr(r)`) density-density
interaction, in the same convention as `Vinteraction`/`VJinteraction`.
Unlike the spin-channel functions, they take the interaction directly as
parameters instead of reading it from `h.V`, so no mean-field convergence is
needed first: they dress the bare susceptibility of whatever Hamiltonian is
passed in, which can also be an already-converged one if the RPA response
about that reference state is wanted

```python
import numpy as np
from pyqula import geometry
g = geometry.chain()
h = g.get_hamiltonian(has_spin=True) # 1D chain at half filling
qs,ws,gammas = h.get_plasmon_bands(V1=0.6,qpath=[[0.3,0.,0.],[0.4,0.,0.],[0.5,0.,0.]],nq=3,
                                    energies=np.linspace(0.,1.,100),delta=2e-2,nk=2000)
```

`qs`, `ws` and `gammas` follow the convention of `get_magnon_bands`. What
matters here is the wavevector: a 1D chain at half filling has perfect
Fermi-surface nesting at $q=\pi$, which strongly enhances the static charge
susceptibility there, so a repulsive `V1` drives the charge channel towards
an instability at the zone boundary, the charge-channel analog of the Stoner
instability of the spin channel above.
