# Spin susceptibility, RPA and magnons

In [Charge response and RKKY](charge-response.md) we computed the bare
response of non-interacting electrons. Now we add the interactions, in the
random phase approximation (RPA), and we find the collective modes of an
interacting state: magnons in a magnet and plasmons in the charge channel.
The interacting state is a mean-field Hamiltonian, obtained as in
[Mean-field Hubbard models](mean-field.md).

## Spin susceptibility in the RPA

When we perturb an interacting system, the perturbation moves some electrons,
this changes the mean field that the other electrons feel, and this moves
more electrons. Summing this series gives the RPA response,

$$
\chi_{\rm RPA}(q,\omega) = \chi_0(q,\omega)\,\left[1 - V(q)\,\chi_0(q,\omega)\right]^{-1},
$$

where $\chi_0$ is the bare response matrix of the mean-field Hamiltonian,
with one entry per site and per operator, and $V(q)$ is the interaction
vertex in the same channel. For the spin response the vertex is built from
the Hubbard $U$ stored on the mean-field Hamiltonian. In a magnetically
ordered state, the RPA response has the magnons, which the bare response does
not have. It is computed with

- `h.get_spinchi_ladder()`: the transverse response, between $S^+$ and $S^-$,
  where the magnons appear.
- `h.get_spinchi_full()`: the full tensor between $S_x$, $S_y$ and $S_z$.

We converge an antiferromagnetic chain and compute its transverse response at
a small momentum transfer:

```python
from pyqula import geometry
import numpy as np
g = geometry.bichain() # two sites per cell, so Neel order fits in the cell
h = g.get_hamiltonian(has_spin=True)
seed = h.copy() ; seed.add_antiferromagnetism(0.5) # symmetry-breaking seed
hmf = h.get_mean_field_hamiltonian(U=3.0,filling=0.5,mf=seed,nk=100) # magnetic state
(es,chis) = hmf.get_spinchi_ladder(energies=np.linspace(0.,2.,400),q=[0.1,0.,0.],nk=40,delta=2e-2)
```

The imaginary part of `chis` has a sharp peak at the magnon energy at this
$q$. This is what inelastic neutron scattering measures in a magnet. Without
spin-orbit coupling the peak goes to zero energy at $q=0$: it is the
Goldstone mode of the broken spin-rotation symmetry.

The keywords are

- `q`: the momentum transfer, in units of the reciprocal lattice vectors.
  Without `q`, the response is dressed at every $q$ of the k-mesh and then
  averaged, which gives the local response, the one a probe on a single site
  sees.
- `RPA`: `True` (default) dresses the response with the interaction; `False`
  gives the bare response.
- `energies`, `delta`, `nk`: frequencies, broadening and k-mesh, as for
  `h.get_chi()`.

Two related methods use the same response:

- `h.get_qdos_iets()` scans `get_spinchi_full` along a q-path, so it gives
  the dispersion of the spin excitations along high-symmetry directions.
- `h.get_iets_ldos()` computes the local response at one energy on every
  site, which is the map of the spin-flip signal that inelastic tunneling
  spectroscopy (IETS) measures with an STM. Together with `h.get_ldos()` it
  gives the elastic and inelastic STM maps.

The q-path labels `"G","K","M"` need a two-dimensional lattice. On the
antiferromagnetic honeycomb lattice, the dispersion along the path is

```python
g2 = geometry.honeycomb_lattice() # already two sublattices per cell
h2 = g2.get_hamiltonian(has_spin=True)
hmf2 = h2.get_mean_field_hamiltonian(U=3.0,filling=0.5,mf="antiferro")
qdisp = hmf2.get_qdos_iets(energies=np.linspace(0.,2.,250),qpath=["G","K","M"],nq=120,nk=60,delta=1e-2)
```

Before reading the spectrum, check that the mean-field state is really
ordered, with `hmf.get_vev("sz")`. An RPA calculation on a non-magnetic state
runs without problems, but it has no magnons. Two things decide whether the
state is ordered. First, the unit cell must be able to hold the order: a
`geometry.chain()` has one site per cell and cannot have Néel order, while
`bichain` and `honeycomb_lattice` have two sublattices. Second, $U$ must be
larger than the critical value: on the honeycomb lattice at half filling
$U=2$ gives a moment of 0.002, so no order, and $U=3$ gives 0.44. The cost of
`get_qdos_iets` grows with `nq`, with the number of energies and with the
number of k-points, $n_k^2$ in two dimensions: the grid above takes about
five minutes.

![Left: transverse spin response of the antiferromagnetic chain at one momentum, with and without RPA. Right: spin response of the antiferromagnetic honeycomb lattice along a q-path, as a heat map.](img/spin_rpa.png#only-light)
![Left: transverse spin response of the antiferromagnetic chain at one momentum, with and without RPA. Right: spin response of the antiferromagnetic honeycomb lattice along a q-path, as a heat map.](img/spin_rpa-dark.png#only-dark)

On the left is the first snippet, the chain at $q=0.1$, with the bare
response (`RPA=False`) added as a reference. Both are summed over the two
sites of the cell and plotted on a log scale. The RPA response has one sharp
peak, the magnon, at $\omega=0.39t$. The bare response has no peak there: it
has only the electron-hole continuum, which starts at the mean-field gap
$1.86t$ and peaks at $2t$, with a tail below the gap from the broadening.
With `nk=80` instead of 40 the two curves do not change. On the right is
`qdisp` of the honeycomb lattice, also on a log color scale. The bright line
is the magnon. It starts at zero at $\Gamma$, the Goldstone mode, rises to
$1.25t$ along $\Gamma$-K, is at $1.16t$ at K and at $1.45t$ at M. The weaker
signal above it is the electron-hole continuum. It starts at about $1.33t$ at
$\Gamma$ and at K, and its lower edge is a bright line. The magnon is the
same with `nk=30`, 40 and 60; the larger `nk` only makes the continuum
smoother, so we use `nk=60`.

??? example "Code of this figure (documentation/figures/spin_rpa.py)"

    ```python
    --8<-- "spin_rpa.py"
    ```

See `examples/1d/rpa/main.py` (RPA spin response against $q$ for an
antiferromagnetic chain), `examples/2d/rpa_triangular/main.py` and
`examples/2d/rpa_honeycomb/main.py` (dispersion along a q-path), and
`examples/0d/rpa_island/main.py` and `examples/0d/rpa_finite_chain/main.py`
(real-space IETS maps) for runnable scripts, and the notebooks
`09_rpa_beyond_onsite.ipynb` and `10_rpa_collective_modes.ipynb` in
`jupyter-notebooks/functionalities/interacting_mean_field_hamiltonians/`.

### Precision on a GPU

The RPA functions can run on a GPU, after `gpu.set_gpu(True)` (see
[GPU](gpu.md)). On a GPU the sum of the bare response is done in single
precision by default, which changes the bare response by about $10^{-7}$.
Close to a pole, where $1-V\chi_0$ is almost singular, the RPA amplifies this
error: next to the Goldstone mode of a Néel state the dressed response
changes by a few parts in $10^4$. If you need the response at a pole with more
digits, use `chi_prec="double"`:

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

On the CPU the default is already double precision.

## Collective modes and magnon bands

The RPA response diverges when the matrix $1-V(q)\chi_0(q,\omega)$ becomes
singular. The frequencies where this happens are the collective modes:
magnons in the spin channel, plasmons in the charge channel. If an eigenvalue
of the kernel crosses zero already at $\omega=0$, the state is unstable (a
Stoner or charge-order instability). `h.get_rpa_kernel_poles()` scans a
frequency window at fixed $q$ and returns all the poles, for any pair of
operators `A`, `B` and any interaction matrix `V`. By default it uses the
charge channel and $q=0$:

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

`V` must have the size of the response matrix. In the charge channel there is
one entry per site, so `V` is $N\times N$. It is not `hmf.V`, the mean-field
interaction in the space of sites and spins ($2N\times2N$), which gives a
dimension error here. For the spin channel use `h.get_magnon_bands()` below,
which builds the vertex from `hmf.V` itself.

`poles` is an array of shape `(npoles,2)`, one row per mode: the frequency of
the pole and the imaginary part of the kernel eigenvalue at the crossing. This
imaginary part can have either sign, so look at its absolute value: a small
value means a sharp mode, a large value means a damped mode or numerical
noise. The snippet above gives three poles, between $3.57t$ and $3.88t$, all
with an absolute value between 2.3 and 2.7, so they are all damped: in the
charge channel this state has no sharp collective mode at $q=0.1$.

### Magnon bands

For the magnons, `h.get_magnon_bands()` finds the poles in the spin channel,
between $S_x$, $S_y$ and $S_z$, with the interaction of the converged mean field, along a
q-path. The result is the magnon dispersion of the ordered state:

```python
qs,ws,gammas = hmf.get_magnon_bands(nq=200,energies=np.linspace(0.01,3.,600),delta=2e-2,nk=80)
```

Each q-point can have a different number of poles, so `qs`, `ws` and `gammas`
are flat arrays of the same length, ready for a scatter plot. `qs` is the
index of the q-point along the path, as in `h.get_bands()`, `ws` the
frequency of the pole and `gammas` its imaginary part; keep
`np.abs(gammas) < threshold` to see only the sharp branches. For the
antiferromagnetic chain you see an acoustic branch that goes to zero at
$q=0$, the Goldstone mode, and an optical branch above it.

![Magnon bands of the antiferromagnetic chain: sharp poles and damped poles of the RPA kernel against momentum.](img/spin_magnons.png#only-light)
![Magnon bands of the antiferromagnetic chain: sharp poles and damped poles of the RPA kernel against momentum.](img/spin_magnons-dark.png#only-dark)

Here we keep the poles with $|\gamma|<0.05$ as sharp. The acoustic branch
goes from zero at $q=0$ up to $1.28t$ at $q=\pi$, and back to zero at $2\pi$.
Above $2.06t$ the gray points are damped poles, the electron-hole continuum.
On a finite k-mesh the continuum is a set of discrete lines, and the poles
between them make the regular pattern of gray points; a finer `nk` makes it
denser, not smoother. The optical branch is the line of sharp points from
$2.35t$ to $2.74t$, between $q=\pi/2$ and $3\pi/2$, inside the continuum. The
two branches are converged: with `nk=40` instead of 80 they move by less
than $10^{-3}t$. The scan takes about half a minute.

??? example "Code of this figure (documentation/figures/spin_magnons.py)"

    ```python
    --8<-- "spin_magnons.py"
    ```

See `examples/1d/magnon_bands/main.py` for a runnable script.

### Interactions beyond onsite

The interaction `V` of `get_rpa_kernel_poles` can also have a range. It is
then given as a dictionary `{(n1,n2,n3): matrix}`, with the key the lattice
vector between the two unit cells, as for the hoppings in
`h.get_hopping_dict()`. It is Fourier transformed in the same way as a
hopping,

$$V(q) = \sum_{\mathbf R} V(\mathbf R)\, e^{2\pi i\, q\cdot \mathbf R},$$

with $q$ in units of the reciprocal lattice vectors. For a first-neighbor
$V_1$ on a chain this gives $V(q) = 2V_1\cos 2\pi q$, repulsive at $q=0$ and
attractive at the zone boundary $q=1/2$:

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

By default this is the charge channel, so `V` here is a density-density
interaction, one entry per site. At $q=1/2$ the snippet gives three poles, at
$0.011t$, $0.081t$ and $0.106t$. The lowest one, almost at zero frequency and
with a small imaginary part (0.03), is the soft charge mode: the half-filled
chain with this $V_1$ is close to a charge-density-wave instability.

### Which magnon method

In the spin channel, an interaction between different sites needs more care.
We write the interaction as $H_{\rm int} = \tfrac12\sum_{ij} V_{ij}\, n_i n_j$.
In the transverse channel, the electron-hole pair $(i,j)$ with the electron on
site $i$ and the hole on site $j$ is scattered by $-V_{ij}$, the Fock
(exchange) term, so the vertex is diagonal in the pair index $(i,j)$, not in
the site index. An onsite $U$ only has pairs with $i=j$, and then a vertex
with one index per site is exact. A neighbor interaction $V_{ij}$ acts on
pairs with $i\neq j$, which a vertex with one index per site does not
contain. For example, a ferromagnetic chain ordered only by $V_1$ would have
no magnon at all with a site vertex. For this reason the spin response
functions (`get_spinchi_full`, `get_spinchi_ladder`, `get_iets_ldos`,
`get_magnon_bands`) use the site vertex only for an onsite $U$, and sum the
series in the pair basis for any interaction between different sites. The
call is the same in both cases. The pair basis does not support Nambu
Hamiltonians.

`h.get_magnon_bands()` takes a `method` keyword with three options:

| interaction | `"rpa"` | `"pair"` | `"tdhf"` |
|---|---|---|---|
| onsite Hubbard $U$ | yes | yes | yes |
| neighbour-shell density-density $V_1,V_2,\dots$ | through the pair basis | yes | yes |
| exchange $J_1,J_2,\dots$ (isotropic or anisotropic) | through the pair basis | yes | yes |
| metallic reference | yes | yes | with `metal=True` |
| non-collinear (canted, spiral) state | yes | yes | yes |
| frequency-resolved $\chi(\omega)$ | yes | yes | no (an eigenproblem) |

- `"rpa"` (default): poles of the RPA kernel on a frequency grid. Exact for
  an onsite $U$; for anything else it uses the pair basis.
- `"pair"`: the same series in the pair basis. Only the pairs that the
  interaction connects enter, so a short-range interaction is cheap. It works
  for metals, and for any spin structure.
- `"tdhf"`: time-dependent Hartree-Fock, an eigenvalue problem with no
  frequency grid and no broadening (see below).

`"pair"` and `"tdhf"` agree for every interaction in the table, and for an
onsite $U$ `"rpa"` gives the same magnons. This is true also at $Q=\Gamma$.
There the Goldstone mode has zero energy, and `"pair"` and `"rpa"` find it at
an energy equal to the broadening `delta`, because the broadening moves it
there. The list of `"rpa"` also has poles of the longitudinal $S_z$ response,
because it uses the full $(S_x,S_y,S_z)$ response. They are close to the ones
that `"pair"` gives with `channel="all"`. For a
collinear state `"pair"` keeps only the spin-flip pairs, so it gives the
magnons and not the charge and longitudinal excitations. With
`channel="all"` it keeps all the pairs and gives those excitations too.
As an example, the honeycomb lattice with
an onsite $U$ and a first-neighbor $V_1$, its magnons and its transverse
response:

```python
import numpy as np
from pyqula import geometry
from pyqula.meanfield import VJinteraction
nk = 24                                 # the SCF and the magnon share this mesh
g = geometry.honeycomb_lattice()
hmf = VJinteraction(g.get_hamiltonian(),U=3.0,V1=0.5,filling=0.5,
                     mf="antiferro",nk=nk,mix=0.5,maxerror=1e-10).hamiltonian
qs,ws,gammas = hmf.get_magnon_bands(method="pair",nq=90,
                    energies=np.linspace(1e-3,3.,400),delta=1e-3,nk=nk)
es,chi = hmf.get_transverse_spinchi(energies=np.linspace(0.,2.,300),
                    q=[0.1,0.,0.],delta=1e-2,nk=nk)   # (Sx,Sy,Sz) x site chi(w)
```

The kernel has an exchange (Fock) term and a Hartree term. For a collinear
state the Hartree term does not enter the transverse response. For a
non-collinear state the spin sectors mix, and without the Hartree term the
spin-rotation symmetry is broken and the acoustic magnon gets a wrong gap. Both
terms are kept, so the Goldstone mode is at zero also for canted and spiral
states.

An exchange interaction $J\,\vec S_i\cdot\vec S_j$ is not a density-density
interaction. Its Ising part $J S^z_iS^z_j$ is, but its transverse part
$\tfrac{J}{2}(S^+_iS^-_j+{\rm h.c.})$ flips a spin on each site. The mean
field writes $S^x_iS^x_j$ and $S^y_iS^y_j$ as Ising terms in two rotated spin
frames, and records the three spin channels on the Hamiltonian it returns.
`"pair"` and `"tdhf"` read these channels, so they include the transverse part,
and for an isotropic $J$ the Goldstone mode is at zero. This needs a
Hamiltonian from `VJinteraction`, or from `h.get_mean_field_hamiltonian()`
with `J1`, `J2`, `J3` or `Jr`. The single-axis couplings `SzSz`, `SxSx` and
`SySy` are recorded as an anisotropic exchange. A global spin rotation keeps
an isotropic exchange, but would turn an anisotropic one into cross terms like
$S^x_iS^y_j$, so `h.global_spin_rotation()` refuses it (unless the rotation is
around the axis of a uniaxial exchange); rotate the Hamiltonian before the
mean-field calculation instead.

![Transverse spin response of the honeycomb antiferromagnet with U and V1 along the q-path, as a heat map, with the poles of the pair-basis magnon bands on top.](img/spin_pair.png#only-light)
![Transverse spin response of the honeycomb antiferromagnet with U and V1 along the q-path, as a heat map, with the poles of the pair-basis magnon bands on top.](img/spin_pair-dark.png#only-dark)

For the map we call `hmf.get_transverse_spinchi()` at 90 q-points of the
default path, which goes from next to $\Gamma$ through K, M and K' back to
$\Gamma$, and plot the trace of its imaginary part on a log color scale. The
circles are the lowest pole from `get_magnon_bands` at each q-point, and the
small gray dots are all the other poles. The lowest bright line is the
magnon: $0.11t$ at the first point of the path, $0.89t$ at K and $1.67t$ at
M, and the circles sit on it. At $q=(0.1,0,0)$, the call of the snippet, the
magnon peak of the response is at $0.57t$. At $\Gamma$, on the right edge,
the response is large at $\omega\to0$, the Goldstone mode, and the pole scan
lists it at $\omega=10^{-3}$, the broadening `delta`. Above the magnon is
the electron-hole continuum, which starts at about $0.9t$ to $1t$ near $\Gamma$ and K. On a
finite k-mesh it is a set of discrete lines, and the interaction shifts each
line that it couples into a pole, so the scan lists between 60 and 130 poles
at each q-point, the gray dots. They are not other collective modes. A
bare line that the interaction does not couple is not a pole: there the
kernel goes through infinity, not through zero, so it is not listed.

The mean field and the response use `nk=24`. For this state the error of the
self-consistent loop goes down slowly with the default mixing `mix=0.3`, so we
take `mix=0.5`. Then `maxerror=1e-10` is reached in about 200 iterations.
With `nk=18` the magnon moves by at most $0.015t$, near K, and with `nk=30` it is at $0.89t$ at K too. The coarse
mesh `nk=6` gives a magnon that is wrong by up to $0.4t$ and a continuum of
a few separate lines. The figure takes about five minutes.

??? example "Code of this figure (documentation/figures/spin_pair.py)"

    ```python
    --8<-- "spin_pair.py"
    ```

See
`jupyter-notebooks/functionalities/interacting_mean_field_hamiltonians/19_magnons_pair_ladder.ipynb`
for an executed notebook that compares the pair basis with time-dependent
Hartree-Fock.

### Magnons from time-dependent Hartree-Fock

You can think of a magnon as an exciton of a mean-field state: a bound
electron-hole pair, where the electron and the hole have opposite spin. So the
Bethe-Salpeter equation of [Excitons](excitons.md), restricted to the
spin-flip pairs, describes it. This is `method="tdhf"`, time-dependent
Hartree-Fock. It works in the pair basis, so it includes interactions between
neighbors:

```python
import numpy as np
from pyqula import geometry
from pyqula.meanfield import VJinteraction
nk = 24                                 # the SCF and the magnon share this mesh
g = geometry.honeycomb_lattice()
scf = VJinteraction(g.get_hamiltonian(),U=3.0,V1=0.5,filling=0.5,
                     mf="antiferro",nk=nk,mix=0.5,maxerror=1e-10)
hmf = scf.hamiltonian
print(hmf.get_goldstone_residual(nk=nk))          # 3e-10: zero to the SCF tolerance
qs,es = hmf.get_magnon_bands(method="tdhf",nk=nk,nq=90,n=2)
```

`qs` and `es` are flat arrays, as for `"rpa"`, with `qs` the index along the
q-path. Without `qpath`, the path on the honeycomb lattice is the line from
$(0,0)$ to $(1,1)$ in units of the reciprocal lattice vectors, so it goes
$\Gamma$, K, M, K', $\Gamma$, and its first point is next to $\Gamma$, not
$\Gamma$ itself. `es` is complex when the mean-field state is unstable against some
excitation. `h.get_magnon_energies(Q=...)` gives the magnon energies at a
single momentum.

The Goldstone theorem is the check. A magnetic state without spin-orbit
coupling breaks the spin-rotation symmetry, so a uniform rotation of all the
spins costs no energy, and there is a magnon at exactly zero energy at $Q=0$.
`h.get_goldstone_residual()` measures the distance from this. It is
proportional to the tolerance of the mean-field calculation, about $3\times10^{-10}$
for `maxerror=1e-10`, so check it before reading any dispersion. A small
imaginary part of the same size on the acoustic branch is expected.

Three things matter:

- The mean field and the magnons must use the same `nk`. A mean field
  converged at `nk=20` is not self-consistent on a mesh of `nk=4`, and then
  the acoustic magnon gets a large wrong gap. The Hamiltonian does not keep
  the mesh it was converged on, so check with `get_goldstone_residual`.
- A metal needs `metal=True`. By default the same number of bands is taken
  as occupied at every k-point, which is wrong for a metal. With `metal=True`
  the occupied states are chosen at each k-point, which is what an itinerant
  magnet needs; for an insulator it changes nothing. In a metal the magnon is
  inside the Stoner continuum and is not the lowest mode, so use
  `by="weight"` to select the branches by their spin weight instead of by
  energy. Also, $E(q)=E(-q)$ only if the occupied states are symmetric under
  $k\to-k$ on the mesh, so choose a mesh with this symmetry.
- An isotropic exchange (`J1`, `J2`, `J3`, `Jr`) from the mean field has its
  transverse part and its Goldstone mode. A hand-built exchange matrix has
  only the Ising part and no record of its channels, so it raises
  `ValueError` (with it, the magnon would get a wrong gap of order $J$). An
  anisotropic exchange (`J1x`, `J1y`, `J1z` not equal, or `SzSz`, `SxSx`,
  `SySy`) breaks the spin-rotation symmetry, so its magnon gap is physical
  and there is no Goldstone mode to check; it also raises `ValueError`, and
  `check_su2=False` computes its spectrum.

For a collinear state only the spin-flip pairs are needed, which is exact and
cheaper, and this is the default. For a canted or spiral state all the pairs
are used automatically (`channel="all"` forces it). This includes the
120-degree spiral of the triangular-lattice Hubbard model, whose Goldstone mode
is at zero to the tolerance of the mean field. For an onsite $U$, where both
are exact, `"tdhf"` and `"rpa"` give the same magnons.

![The two lowest magnon branches of the honeycomb antiferromagnet with U and V1, from time-dependent Hartree-Fock, along the q-path.](img/spin_tdhf.png#only-light)
![The two lowest magnon branches of the honeycomb antiferromagnet with U and V1, from time-dependent Hartree-Fock, along the q-path.](img/spin_tdhf-dark.png#only-dark)

The Goldstone residual is $3\times10^{-8}$, so the dispersion can be read.
The path starts next to $\Gamma$, not at it, and ends at $\Gamma$. There the
lowest mode is at zero, the Goldstone mode, and the second one is at
$0.91t$. At K the lowest mode is at $0.89t$, and at M the two modes are at
$1.67t$ and $2.41t$. The lowest mode has the same energies as the lowest
poles of the pair-basis map above, to $10^{-3}t$, as it should be. Both
branches are smooth along the path with `nk=24`. On a coarse mesh, such as
`nk=6`, they zigzag between neighboring $Q$, because the electron-hole pairs
of the mesh at $k$ and $k+Q$ change from one $Q$ to the next. The lowest mode
is converged: it moves by at most $0.015t$ from `nk=18` to 24, and by
$0.005t$ at K from `nk=24` to 30. The second mode near K and K' is the bottom
of the electron-hole continuum, not a second magnon, and it converges more
slowly: at K it is at $1.09t$, $1.01t$ and $0.97t$ for `nk=18`, 24 and 30.
Its limit is the bottom of the continuum at K, the smallest
$E_c(\mathbf k+\mathbf Q)-E_v(\mathbf k)$ of the mean-field bands, which is
$0.91t$, the mean-field gap, since the valence band has its maximum at K and
the conduction band its minimum at K'. On a finite mesh the lowest pair state
sits above this edge and comes down to it only slowly as `nk` grows, so the
second mode is a discrete level of the continuum and not a converged branch.
The magnon at K, $0.89t$, is just below the edge.
The figure takes about eight minutes.

??? example "Code of this figure (documentation/figures/spin_tdhf.py)"

    ```python
    --8<-- "spin_tdhf.py"
    ```

See `examples/2d/magnon_bands_tdhf/main.py` for a runnable script, and
`jupyter-notebooks/functionalities/interacting_mean_field_hamiltonians/18_magnons_tdhf.ipynb`
for an executed notebook.

### Chern number of a magnon band

We will now see how to compute the Chern number of a magnon band, with
`h.get_magnon_chern()`. For an electron band the Chern number counts the
winding of the Bloch state over the Brillouin zone. What plays the role of
the Bloch state for a magnon? In time-dependent Hartree-Fock a magnon with
momentum $\mathbf Q$ is a sum of electron-hole pairs, an electron in an empty
band $c$ at $\mathbf k+\mathbf Q$ and a hole in an occupied band $v$ at
$\mathbf k$, with the amplitudes $X_{vc}(\mathbf k)$. The pairs are built
again at every $\mathbf Q$, so we write the magnon in the orbitals instead,

$$
\Phi_{\mathbf Q}(\mathbf k;\alpha,\beta)=\sum_{vc}X_{vc}(\mathbf k)\,
u_{c,\mathbf k+\mathbf Q}(\alpha)\,u^*_{v,\mathbf k}(\beta) ,
$$

where $u_{n,\mathbf k}(\alpha)$ is the Bloch state of band $n$ on the orbital
$\alpha$, spin included. This is the amplitude of the operator
$c^\dagger_{\alpha,\mathbf k+\mathbf Q}c_{\beta,\mathbf k}$ in the magnon, so
it does not depend on the phases of the Bloch states. A magnon with a
de-excitation part $Y$ gets a second block, built in the same way. The Chern
number of a group of magnon bands is then

$$
C=\frac{1}{2\pi}\sum_{\square}F_{\square} ,
$$

where $F_\square$ is the Berry flux of $\Phi_{\mathbf Q}$ through one
plaquette of a mesh of $\mathbf Q$, from the overlaps of the states at its
four corners. The sign is the one of `h.get_chern()`. The magnon bands are
what inelastic neutron scattering measures, and a band with a Chern number
gives chiral magnon states at the edges of the magnet.

We take a ferromagnet on the honeycomb lattice, with a second-neighbor
hopping $t'=0.3t$, a Kane-Mele spin-orbit coupling $\lambda=0.15t$ and a
Zeeman field $B=3t$, with $U=8t$ at half filling. The field polarizes the
magnet along $-z$ and gives the magnons a gap, and the product of $\lambda$
and $t'$ plays the role of a Dzyaloshinskii-Moriya interaction:

```python
from pyqula import geometry
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(tij=[1.0,0.3]) # first and second neighbor hopping
h.add_kane_mele(0.15) # spin-orbit coupling, conserves Sz
h.add_zeeman([0.,0.,3.0]) # a field, so that the magnon has a gap
nk = 8 # the mean field and the magnons share this mesh
hmf = h.get_mean_field_hamiltonian(U=8.0,filling=0.5,mf="ferro",nk=nk)
c0 = hmf.get_magnon_chern(bands=[0],nq=8,nk=nk) # the lower magnon band
c1 = hmf.get_magnon_chern(bands=[1],nq=8,nk=nk) # the upper magnon band
c01 = hmf.get_magnon_chern(bands=[0,1],nq=8,nk=nk) # the two together
```

The magnet is saturated, `hmf.get_vev("sz")` is $-1$ on both sites. Each
result is a float that is an integer up to roundoff: `c0` is $+1$, `c1` is
$-1$ and `c01` is $0$. With `add_kane_mele(-0.15)` the two signs flip. The
integers are the same with `nq=12`, and with a mean field and magnons at
`nk=12`. The snippet takes about 12 seconds.

The number alone does not say if it can be trusted. With `full_output=True`
the method returns a record instead:

```python
r = hmf.get_magnon_chern(bands=[0],nq=8,nk=nk,full_output=True)
print(r.chern,r.gap,r.bound,r.max_flux) # the integer and what it rests on
```

`r.energies` has the energies of the band at the points `r.qs` of the mesh,
from $4.55t$ to $4.80t$ here. `r.gap` is $0.33t$, the smallest distance to
the other magnon band on the mesh. `r.bound` is $2.75t$, the smallest
distance below the Stoner continuum, the electron-hole pairs with a spin
flip. `r.max_flux` is $0.54$, about $0.17\pi$, the largest Berry flux of one
plaquette, and `r.fluxes` has all of them.

![Berry curvature of the lower and the upper magnon band of the honeycomb ferromagnet over the Brillouin zone, from the fluxes of the record on a mesh of 12 by 12 points.](img/spin_magnonchern.png#only-light)
![Berry curvature of the lower and the upper magnon band of the honeycomb ferromagnet over the Brillouin zone, from the fluxes of the record on a mesh of 12 by 12 points.](img/spin_magnonchern-dark.png#only-dark)

The figure takes `nq=12` and shows `r.fluxes` divided by the area of a
plaquette, the Berry curvature, whose mean over the zone is $C$. The
curvature of the lower band is positive and sits around K and K', at
$(1/3,1/3)$ and $(2/3,2/3)$ in units of the reciprocal lattice vectors,
with a largest value of $7.3$. This is where the two magnon bands come
closest, $0.26t$ apart on this mesh. The upper band has the opposite
curvature at the same points, so the two together give zero. The largest
flux of a plaquette is $0.32$, about $0.10\pi$, so the mesh resolves the
bands.

??? example "Code of this figure (documentation/figures/spin_magnonchern.py)"

    ```python
    --8<-- "spin_magnonchern.py"
    ```

The integer is the Chern number of the band only under four conditions:

- The band is separated from the other magnon bands at every $\mathbf Q$,
  and it is below the Stoner continuum. A band that touches another one at a
  point of the mesh raises `ValueError`. A touching between two points of
  the mesh is not seen, also not in `r.gap`, which is measured on the points
  of the mesh, so a second `nq` is the check. The default `nq=12` holds
  $\Gamma$, M and K of a hexagonal zone, where a symmetry can force two
  bands to touch.
- The magnons are converged in `nk`, the mesh of the mean field and of the
  magnon problem, and the integer does not change with `nq`.
- The Berry flux of every plaquette is below $\pi$. A flux close to $\pi$ in
  `r.max_flux` means that `nq` has to grow. A group of $n$ bands that share
  one state, as a spin triplet, has $n$ times the flux of one band.
- The mean-field state is stable. A complex or a negative magnon energy
  raises `ValueError`. A magnet with no gap at $Q=0$ and no net moment, an
  antiferromagnet without a field, needs `shift=`, a mesh that avoids $Q=0$:
  its Goldstone mode has no eigenvector there, and whether the mesh with
  $Q=0$ raises depends on the sign of the small residual of the mean field.

The keywords are:

- `bands`: the group of bands, counted from the lowest magnon, a contiguous
  list; `[0]` by default.
- `nq`: the points of the mesh of $\mathbf Q$ along each reciprocal lattice
  vector, 12 by default.
- `shift`: the offset of the mesh, in units of the reciprocal lattice
  vectors.
- `nk`, `V`, `channel`, `metal`: as in `h.get_magnon_energies()`.
- `full_output`: return the record instead of the number.

See `examples/2d/magnon_chern/main.py` for a runnable script, and
`jupyter-notebooks/functionalities/topological_characterization/15_magnon_exciton_chern.ipynb`
for an executed notebook.

## Charge response and plasmons

The same RPA can be done in the charge channel, with
`h.get_densitychi_RPA()` and `h.get_plasmon_bands()`. These are the
density-density versions of `get_spinchi_full` and `get_magnon_bands`. The
collective modes are now plasmons, or the modes of a charge order. The
interaction is a density-density interaction with an onsite `U`, neighbor
shells `V1`, `V2`, `V3`, and a general `Vr(r)`, as in `Vinteraction` and
`VJinteraction`. Different from the spin channel, the interaction is given as
keywords and is not read from `h.V`, so no mean-field calculation is needed:
the bare response of the Hamiltonian you give is dressed with this
interaction. This Hamiltonian can also be a converged mean-field one.

```python
import numpy as np
from pyqula import geometry
g = geometry.chain()
h = g.get_hamiltonian(has_spin=True) # 1D chain at half filling
qs,ws,gammas = h.get_plasmon_bands(V1=0.6,qpath=[[0.3,0.,0.],[0.4,0.,0.],[0.5,0.,0.]],nq=3,
                                    energies=np.linspace(0.,1.,100),delta=2e-2,nk=2000)
```

`qs`, `ws` and `gammas` are as in `get_magnon_bands`. What matters here is the
wavevector. At half filling the chain has perfect nesting at $q=1/2$
($k=\pi$), so the static charge response is large there. A repulsive `V1`
gives $V(q)=2V_1\cos2\pi q<0$ at the zone boundary, so it pushes the charge
channel toward an instability at $q=1/2$, a charge-density wave. This is the
charge version of the Stoner instability of the spin channel.

![Imaginary part of the RPA charge response of the chain with a first-neighbor V1, over momentum transfer and frequency, with the edges of the electron-hole continuum.](img/spin_plasmon.png#only-light)
![Imaginary part of the RPA charge response of the chain with a first-neighbor V1, over momentum transfer and frequency, with the edges of the electron-hole continuum.](img/spin_plasmon-dark.png#only-dark)

The map is not in the snippet: we call `h.get_densitychi_RPA()` with the
same `V1=0.6` at 201 values of $q$ with `nk=1000`, and plot it on a log
color scale. With `nk=3000` the map is the same. The dashed lines are the
edges of the free electron-hole continuum, $2t\sin 2\pi q$ and $4t\sin\pi q$.
The circles are the poles of `get_plasmon_bands` at 50 values of $q$, with
`energies` from 0 to $4t$. At small $q$, where $V(q)>0$, the RPA moves the
weight to the upper edge and a little above it: this is the plasmon. Its
poles go from $0.28t$ at $q=0.02$, just above the edge, to $2.49t$ at
$q=0.21$, where it enters the continuum and the poles stop. Their $|\gamma|$
is between 0.11 and 0.3 from $q=0.04$ on, and 0.7 at $q=0.02$, so the
plasmon is damped most at the smallest $q$. Near $q=1/2$, where $V(q)<0$, the weight moves to
the lower edge and below it, toward zero frequency: this is the soft charge
mode of the charge-density wave. Its poles go from $1.26t$ at $q=0.39$ down
to $0.32t$ at $q=0.45$, below the edge, and they get sharper as the frequency
goes down, $|\gamma|=0.01$ at $q=0.45$. Closer to $q=1/2$ there is no pole.
At $q=1/2$ the real part of the kernel eigenvalue changes sign at $0.60t$,
but the eigenvalue stays at about 0.6 in absolute value, so the kernel never
comes close to zero. The window of the snippet, `energies` from 0 to 1,
misses the mode at $q=0.4$, at $1.115t$, just below the edge of the
continuum at $1.18t$; with `energies` up to $2t$ the same call finds it. The
figure takes about three and a half minutes.

??? example "Code of this figure (documentation/figures/spin_plasmon.py)"

    ```python
    --8<-- "spin_plasmon.py"
    ```
