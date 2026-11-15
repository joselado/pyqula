# Optical conductivity

A response function tells how a system reacts to a weak perturbation. A
field couples to an operator $B$, we measure the change of an operator $A$,
and the ratio, as a function of frequency $\omega$ and momentum transfer
$q$, is the response function $\chi_{AB}(q,\omega)$. Its imaginary part is
what a spectroscopy measures. The response pages go in three steps: the bare
response of a single-particle Hamiltonian (this page and
[Charge response and RKKY](charge-response.md)), the same response dressed
with interactions in the random phase approximation (RPA), whose poles are
magnons and plasmons ([Spin response and magnons](spin-response.md)), and
the Bethe-Salpeter equation for excitons ([Excitons](excitons.md)).

## The Kubo formula

We start with the response to an electric field, the optical conductivity
tensor $\sigma_{ab}(\omega)$ of a periodic Hamiltonian, computed with
`h.get_optical_conductivity()`. It is given by the Kubo-Greenwood formula,

$$
\sigma_{ab}(\omega) = \frac{i e^2 \hbar}{N_k V_{\rm cell}} \sum_{\mathbf k}
\sum_{n, m} \frac{f_n - f_m}{E_m - E_n}
\frac{v^a_{nm} v^b_{mn}}{\hbar\omega + i\eta - (E_m-E_n)} ,
$$

where $n,m$ are the bands at momentum $\mathbf k$, $E_n$ their energies,
$f_n$ their occupations at temperature $T$, $v^a_{nm}$ the matrix element of
the velocity along direction $a$, $N_k$ the number of k-points, $V_{\rm cell}$
the volume (area, length) of the unit cell, and $\eta$ a broadening. The
physical meaning is simple: light of frequency $\omega$ is absorbed when it
can move an electron from an occupied band to an empty band at energy
$\hbar\omega$ above, and the weight of this transition is given by the
velocity matrix element.

The terms with $E_n=E_m$ give the intraband (Drude) response. For them the
factor $(f_n-f_m)/(E_m-E_n)$ is replaced by its limit $-\partial f/\partial E$,
and the result is a peak at $\omega=0$,

$$
\sigma^{\rm intra}_{ab}(\omega) = \frac{i D_{ab}}{\omega + i\eta},
\qquad
D_{ab} = \frac{1}{N_k V_{\rm cell}} \sum_{\mathbf k,n}
\left(-\frac{\partial f}{\partial E}\right)_{E_n} v^a_{nn} v^b_{nn} ,
$$

with $D_{ab}$ the Drude weight, available as `h.get_drude_weight()`. The
velocity is the current operator of the tight-binding model,
$v_a = i[H, r_a]$, built from the full vector of each bond, also the bonds
inside the unit cell. Because of this the tensor has the symmetry of the
lattice: on the honeycomb lattice $\sigma_{xx}=\sigma_{yy}$ exactly, and
graphene gives its universal absorption $\pi e^2/4h$ per spin.

The full complex tensor is returned. $\mathrm{Re}\,\sigma_{xx}$ is the
optical absorption. $\sigma_{xy}$ is the magneto-optical (Kerr and Faraday)
response. Its $\omega\to0$ limit is the anomalous Hall conductivity, which
for a Chern insulator with Chern number $C$ is quantized to

$$\sigma_{xy}(0) = -C\,\frac{e^2}{h},$$

with $C$ as returned by `h.get_chern()`.

Units: $\hbar=e=1$, energies in units of the hopping $t$, lengths in units of
the lattice constant. The conductivity is returned in units of $e^2/\hbar$
(times $a^{2-d}$ in $d$ dimensions), so one conductance quantum
$e^2/h = e^2/(2\pi\hbar)$ is the value $1/(2\pi)$. The Fermi energy is at
zero, as everywhere in pyqula.

The absolute normalization is fixed by the f-sum rule,

$$\int_{-\infty}^{\infty} \mathrm{Re}\,\sigma_{aa}(\omega)\,d\omega = \pi W_{aa},$$

where $W_{ab}$ is the diamagnetic weight, the occupied average of
$\partial^2 H/\partial k_a\partial k_b$, available as
`h.get_sum_rule_weight()`. The weight is shared between the Drude peak and the
interband absorption.

## A Chern insulator

We take the honeycomb lattice with a Haldane term, a Chern insulator, and
compute its absorption and its Hall response:

```python
import numpy as np
from pyqula import geometry
g = geometry.honeycomb_lattice() # create a honeycomb lattice
h = g.get_hamiltonian(has_spin=False) # get the Hamiltonian
h.add_haldane(0.2) # Chern insulator
h.shift_fermi(0.3) # put the Fermi energy in the gap

(ws,s) = h.get_optical_conductivity(energies=np.linspace(0.,4.,401),
                                    nk=200,T=0.02,delta=0.05)
absorption = s[:,0,0].real # Re sigma_xx, the optical absorption

# the DC Hall response is quantized to the Chern number
(w0,s0) = h.get_optical_conductivity(energies=[0.],nk=40,T=0.01,delta=1e-3)
print("sigma_xy(0) in e^2/h:",2.*np.pi*s0[0,0,1].real) # -1
```

`ws` contains the frequencies and `s` the $3\times3$ tensor at each
frequency, so `s` has shape `(len(ws),3,3)`. The absorption is zero below
the gap and starts at the gap. The second call computes the tensor at zero
frequency with a small broadening, and $2\pi\sigma_{xy}$ comes out as $-1$,
one conductance quantum with the sign of $-C$.

![Real part of the longitudinal and Hall optical conductivity of the Haldane Chern insulator against frequency, with the band gap marked.](img/optical_chern.png#only-light)
![Real part of the longitudinal and Hall optical conductivity of the Haldane Chern insulator against frequency, with the band gap marked.](img/optical_chern-dark.png#only-dark)

The dashed line is the band gap, $1.8t$, from `h.get_gap()`. Below the gap
the absorption $\mathrm{Re}\,\sigma_{xx}$ is almost zero; the small tail
comes from the broadening `delta=0.05`. It rises at the gap and has its
maximum, $6.45\,e^2/h$, at $\omega=2.0t$. The Hall part
$\mathrm{Re}\,\sigma_{xy}$ starts at $-1.00\,e^2/h$ at $\omega=0$, the
quantized value, goes down to $-4.18\,e^2/h$ at the gap, and changes sign
at $\omega=2.03t$, next to the maximum of the absorption. With `delta=0.05`
the k-mesh has to be fine: at `nk=40` both curves have wiggles above the gap,
which are the separate k-points of the mesh. At `nk=200` they are gone, and
`nk=300` changes the curves by less than $2\times10^{-4}\,e^2/h$.

??? example "Code of this figure (documentation/figures/optical_chern.py)"

    ```python
    --8<-- "optical_chern.py"
    ```

The keywords are

- `energies`: the frequencies $\hbar\omega$.
- `nk`: number of k-points per direction. A small `delta` needs a large `nk`,
  otherwise the absorption shows the discrete k-mesh.
- `T`: temperature of the Fermi function.
- `delta`: the broadening $\eta$.
- `intraband`, `interband`: switch on or off the Drude part and the interband
  part.
- `integration`: how the k-integral is done, `"grid"` (the default), or
  `"triangle"` and `"qtci"` for a small broadening, see
  [Small broadening](#small-broadening).

The formula works for one-, two- and three-dimensional Hamiltonians. In
three dimensions $\sigma$ is a bulk conductivity, in units of
$e^2/(\hbar a)$. A stack of decoupled layers with spacing $c$ gives the
sheet conductance of one layer divided by $c$. Superconducting (Nambu)
Hamiltonians raise `NotImplementedError`: the hole block carries the
opposite charge, so the current is not $i[H,r]$.

## Small broadening

We will now see how to compute the absorption with a very small broadening
$\eta$. On a k-mesh, every k-point gives a Lorentzian of width $\eta$ at its
transition energy $E_m(\mathbf k)-E_n(\mathbf k)$. Two neighboring k-points
have transition energies that differ by roughly the velocity of
$E_m-E_n$ times the mesh spacing. When this difference is larger than
$\eta$, the Lorentzians do not overlap, and the spectrum shows every k-point
as a separate spike. For $\eta=10^{-3}t$ on the honeycomb lattice, the
grid sum needs thousands of k-points per direction.

The way we avoid this is by doing the k-integral differently. The
transition energies $E_m-E_n$ and the weights
$\frac{f_n-f_m}{E_m-E_n}v^a_{nm}v^b_{mn}$ are smooth functions of
$\mathbf k$. Only the resolvent $1/(\hbar\omega+i\eta-(E_m-E_n))$ is sharp.
We split the k-mesh into triangles, take the transition energy as linear
inside each triangle, and integrate the resolvent over the triangle exactly,

$$
\frac{1}{A}\int_{\triangle} \frac{d^2k}{z-E(\mathbf k)} =
2\,\psi[E_1,E_2,E_3],
\qquad
\psi(E) = -(E-z)\ln(E-z) + (E-z),
$$

where $z=\hbar\omega+i\eta$, $A$ is the area of the triangle, $E_1,E_2,E_3$
are the transition energies at its corners, and $\psi[E_1,E_2,E_3]$ is the
second divided difference of $\psi$. The weight is taken as its average over
the three corners. In one dimension the triangles are segments. This is the
two-dimensional version of the tetrahedron method of band structure codes.
As a result, the mesh only has to resolve the bands and not the broadening,
and all frequencies come from the same mesh.

We compare the two integrations on a gapped honeycomb lattice, on the same
mesh:

```python
import numpy as np
from pyqula import geometry
g = geometry.honeycomb_lattice() # create a honeycomb lattice
h = g.get_hamiltonian(has_spin=False) # get the Hamiltonian
h.add_sublattice_imbalance(0.3) # gapped honeycomb, gap 0.6

ws = np.linspace(0.,3.5,701) # frequencies
kw = dict(energies=ws,nk=256,T=1e-3,delta=1e-3) # broadening far below the mesh spacing
(ws,sg) = h.get_optical_conductivity(integration="grid",**kw) # sum over the k-mesh
(ws,st) = h.get_optical_conductivity(integration="triangle",**kw) # exact integral over triangles
```

`sg` and `st` are the conductivity tensors of the two integrations, with
the same shape as before.

![Real part of the optical conductivity of a gapped honeycomb lattice at a broadening of 0.001 t, from a plain sum over a 256 by 256 k-mesh, which is a set of spikes, and from the triangle integration on the same mesh, which is a smooth curve.](img/optical_triangle.png#only-light)
![Real part of the optical conductivity of a gapped honeycomb lattice at a broadening of 0.001 t, from a plain sum over a 256 by 256 k-mesh, which is a set of spikes, and from the triangle integration on the same mesh, which is a smooth curve.](img/optical_triangle-dark.png#only-dark)

With the grid sum the absorption is a set of spikes, the single k-points
of the mesh. With the triangle integration the curve is smooth. It is zero
below the gap $0.6t$, jumps at the gap, and has a sharp peak at
$\hbar\omega=2.09t$, the van Hove singularity of the transitions at the M
point. Doubling the mesh to `nk=512` changes the triangle curve by less
than $0.01\,e^2/h$, except exactly at the absorption edge. There the
transition energy has its minimum, and the linear interpolation is worst.

??? example "Code of this figure (documentation/figures/optical_triangle.py)"

    ```python
    --8<-- "optical_triangle.py"
    ```

The triangle result converges as $1/n_k^2$. For a smooth integrand, a
broadening larger than the band features or a frequency inside the gap,
the grid sum converges faster, so there `integration="grid"` is the better
choice. The triangle integration removes the mesh limit set by $\eta$, not
the one set by the temperature: in a metal the occupations jump at the
Fermi surface within a width $T$, and the Drude term is computed as in the
grid sum.

With `integration="qtci"` the same triangle integral runs on a mesh of
$2^R\times2^R$ points, with `nk` rounded up to a power of two. The
transition energies and the weights are not computed on every point of
this mesh. They are learned from a small set of k-points with a quantics
tensor cross interpolation, which works because they are smooth, and then
rebuilt on the whole mesh:

```python
(ws,sq) = h.get_optical_conductivity(energies=[1.0,2.0],nk=1024,T=1e-3,
                                     delta=1e-3,integration="qtci")
```

On the gapped honeycomb, the number of diagonalizations stops growing with
the mesh: `nk=4096` needs about $8\times10^4$ of them instead of
$1.7\times10^7$. On the same mesh the result agrees with
`integration="triangle"` to about $10^{-5}$ relative, and to $10^{-3}$ at
the absorption edge. This helps when one
diagonalization is expensive, for a Hamiltonian with many orbitals. For a
two-band model the triangle integration on a plain mesh is already cheap.
Band crossings and the Fermi surface of a metal are not smooth, and they
make the interpolation more expensive.

The keywords of the two integrations are

- `integration`: `"grid"`, `"triangle"` or `"qtci"`.
- `nk`: for `"triangle"`, the mesh that resolves the bands; for `"qtci"`, the
  fine mesh, rounded up to a power of two.
- `delta`: must be positive for `"triangle"` and `"qtci"`.
- `tolerance`: relative accuracy of the interpolation, only for `"qtci"`.

`"qtci"` works for two-dimensional Hamiltonians only; in one dimension
`"triangle"` on a fine mesh is already cheap. In three dimensions
`"triangle"` cuts each cube of the mesh into six tetrahedra and integrates
the resolvent exactly over each one. At $\eta=10^{-2}t$ on a stack of
Chern insulators it needs about 100 times fewer k-points than the grid for
the same accuracy at the absorption edge. Most of the work is done once for
all frequencies, so 400 frequencies cost two to four times as much as 40,
not ten times. A full script is in
`examples/2d/optical_conductivity_triangle/main.py`.

## Large systems

We will now see how to compute the conductivity of a system that is too
large for a band structure: a disordered supercell with $10^4$ to $10^6$
orbitals, or a big flake. The way we do this is by never diagonalizing.
The absorptive part of the tensor, the part that takes energy from the
field, is written with two delta functions of the Hamiltonian,

$$
\sigma^{H}_{ab}(\omega) = \frac{\pi}{V}\int dE\,\frac{f(E)-f(E+\omega)}{\omega}\,
{\rm Tr}\left[v_b\,\delta(E-H)\,v_a\,\delta(E+\omega-H)\right] ,
$$

with $V$ the volume of the supercell. In the eigenbasis this is the
delta-function part of the Kubo formula above. Each delta function is
expanded in Chebyshev polynomials of $H$, as in
[Kernel polynomial method](kpm.md), and the trace is estimated with a few
random vectors. Then the cost and the memory grow linearly with the number
of orbitals. The reactive part follows from the absorption at all
frequencies, by the Kramers-Kronig relation,

$$
\sigma_{ab}(\omega) = \frac{i}{\pi}\int d\omega'\,
\frac{\sigma^{H}_{ab}(\omega')}{\omega+i0^+-\omega'} ,
$$

meaning that the full complex tensor is fixed by the absorption, as it is
in an experiment. This is `integration="kpm"`:

```python
import numpy as np
from pyqula import geometry
g = geometry.honeycomb_lattice().get_supercell(30) # 1800 sites
h = g.get_hamiltonian(has_spin=False,is_sparse=True)
h.add_sublattice_imbalance(0.5) # gap of 1
h.add_onsite(lambda r: 0.6*(np.random.random()-0.5)) # onsite disorder
ws = np.linspace(0.,4.,100) # frequencies
(ws,s) = h.get_optical_conductivity(energies=ws,integration="kpm",
                                    delta=0.05,ntries=4)
```

`s` is the same complex array of shape `(nw,3,3)` as in the band modes,
here at the Gamma point of the supercell. The Fermi function is taken at the
temperature `T`, which is `delta` by default, and any `T` works, also one
much smaller than `delta`.

![Real and imaginary part of the optical conductivity of a disordered gapped honeycomb supercell of 1800 sites against frequency, from the Chebyshev expansion.](img/optical_kpm.png#only-light)
![Real and imaginary part of the optical conductivity of a disordered gapped honeycomb supercell of 1800 sites against frequency, from the Chebyshev expansion.](img/optical_kpm-dark.png#only-dark)

${\rm Re}\,\sigma_{xx}$ is zero below the gap, starts at $\omega\approx0.9$
and has its maximum, $0.38\,e^2/\hbar$, at $\omega=2.2$. The wiggles are
the levels of a finite cell seen with four random vectors.
${\rm Im}\,\sigma_{xx}$ is negative below the gap, as for a dielectric,
with its minimum at the absorption edge, and it changes sign near the
maximum of the absorption. With `kernel="lorentz"` the line of each
transition is the Lorentzian of the band modes, and on a small cell the two
agree to a few percent, for the real and the imaginary part, with the Fermi
level in a gap or in a band. A full script, with the DC conductivities
below, is `examples/2d/kpm_conductivity/main.py`, and an executed notebook is
`jupyter-notebooks/functionalities/chebyshev_kpm_algorithms/06_kpm_conductivity.ipynb`.

??? example "Code of this figure (documentation/figures/optical_kpm.py)"

    ```python
    --8<-- "optical_kpm.py"
    ```

The keywords of this mode are

- `nk=1`: k-points per direction. A large supercell at the Gamma point is
  the usual choice.
- `delta`: the half width of the line of one transition. It sets the number
  of polynomials, and the cost grows as $1/\delta^2$.
- `kernel`: `"jackson"` (default), a Gaussian line, or `"lorentz"`, a
  Lorentzian line.
- `ntries`: the number of random vectors. At $10^5$ orbitals one or two are
  enough, because the trace of a large cell self-averages.
- `trace`: `"exact"` uses every basis vector, for a small system.
- `return_error`: also return the statistical error over the random vectors.
- `reactive`: `False` returns only the absorptive part, at a lower cost.

The DC tensor of a large system is `h.get_dc_conductivity()`. By default it
uses the Kubo-Bastin formula, which keeps the states of the Fermi sea. So
with the Fermi level in the gap of a Chern insulator it gives the quantized
Hall conductivity:

```python
g = geometry.honeycomb_lattice().get_supercell(10) # 200 sites
h = g.get_hamiltonian(has_spin=False,is_sparse=True)
h.add_haldane(0.2) # Chern insulator, C=1
s = h.get_dc_conductivity(fermi=0.,delta=0.1,ntries=2) # 3x3 tensor
print(2*np.pi*s[0,1]) # -C, the Hall conductivity in units of e^2/h
```

This prints $-0.96$, which is $-C$ within the error of two random vectors
on 200 sites. The same plateau is the $\omega\to0$ limit of
${\rm Re}\,\sigma_{xy}(\omega)$ from `integration="kpm"`, which never uses
the Kubo-Bastin formula, so the two are an independent check of each other.

For the longitudinal conductivity of a metal, `mode="singleshot"` uses the
Kubo-Greenwood formula at zero temperature,

$$
\sigma_{aa}(E_F) = \frac{1}{\pi V}\,
{\rm Tr}\left[v_a\,{\rm Im}\,G\,v_a\,{\rm Im}\,G\right] ,
\qquad G = \left(E_F + i\delta/2 - H\right)^{-1} ,
$$

with one Chebyshev series of $G$ for each Fermi level. Each delta function
is a Lorentzian of half width $\delta/2$, so $\delta$ keeps its meaning, and
a clean metal gives the Drude value $D/\delta$. The cost grows as $1/\delta$
and not as $1/\delta^2$, so this mode reaches the small broadening that a
weakly disordered metal needs:

```python
np.random.seed(1)
g = geometry.honeycomb_lattice().get_supercell(40) # 3200 sites
h = g.get_hamiltonian(has_spin=False,is_sparse=True)
h.add_onsite(lambda r: 3.0*(np.random.random()-0.5)) # Anderson disorder
s = h.get_dc_conductivity(fermi=[0.,1.],delta=0.05,mode="singleshot",
                          component="xx",ntries=16,seed=1) # sigma_xx at two E_F
```

This gives $\sigma_{xx}=0.10$ at $E_F=0$ and $0.13$ at $E_F=1$, with an
error of about 10% from the 16 random vectors, and the Kubo-Bastin mode
gives the same numbers. What matters is that the result no longer changes
with $\delta$. At $E_F=1$ the values for $\delta=0.1$, $0.05$ and $0.02$
agree within the error. At the Dirac point $\delta=0.02$ gives $0.07$,
lower by about two error bars, so there the convergence is not settled
with this cell. If the result still changes, the broadening is larger than
the scattering rate, and $\delta$ has to go down, with a larger cell to
keep the level spacing below $\delta$. This mode returns only the diagonal
of the tensor, the off-diagonal entries are NaN, and it does not take a
temperature.

## Second-harmonic generation

We will now see the response at second order in the field. A crystal
without inversion symmetry, driven by light of frequency $\omega$, emits
light at $2\omega$. This is second-harmonic generation (SHG). The current
at $2\omega$ is

$$
J_a(2\omega) = \sigma_{abc}(2\omega;\omega,\omega)\,E_b(\omega)\,E_c(\omega),
$$

and $\sigma_{abc}$ is computed with `h.get_shg_conductivity()`. It is
symmetric in $b$ and $c$. Experiments quote the susceptibility instead,
$P_a(2\omega) = \epsilon_0 \chi_{abc} E_b E_c$. Since $J = dP/dt$, the two
are related by

$$
\chi_{abc} = \frac{i\,\sigma_{abc}}{2(\omega+i\eta)\,\epsilon_0} ,
$$

and `h.get_shg_susceptibility()` returns $\chi_{abc}$ in SI units, in
nm$^2$/V for a two-dimensional system. This is the sheet susceptibility.
Divide it by a thickness to get a bulk value in nm/V.

The response has two kinds of resonances. One appears when $2\hbar\omega$
matches a transition $E_m-E_n$ between an occupied and an empty band, and
the other appears when $\hbar\omega$ matches it. So the SHG spectrum shows
the absorption features of the linear spectrum at half their energy, and
also at their own energy. Every frequency has the same broadening,
$\omega\to\omega+i\eta$, so the $2\omega$ poles have the broadening $2\eta$.

Under inversion $\mathbf r\to-\mathbf r$ the current and the field change
sign, so $\sigma_{abc}$ changes sign too, and it is exactly zero in a
centrosymmetric crystal. This is why SHG is used to count layers in 2H
stacks of transition metal dichalcogenides: an odd number of layers gives a
signal, an even number gives almost none.

The second harmonic is one case of the second-order response. The dc
current under light (the shift and injection currents), the sum and
difference frequencies, the nonlinear transport of a metal and the third
harmonic are in [Nonlinear response](nonlinear-response.md).

The calculation uses the velocity operator of the tight-binding model, the
same as in [The Kubo formula](#the-kubo-formula), and also the second and
third derivatives of $H(\mathbf k)$ with respect to $\mathbf k$. With these
terms the result is exact for the tight-binding model with any number of
bands, and it does not depend on the gauge of the field.

We take the honeycomb lattice with a staggered onsite energy, a two-band
model of hexagonal boron nitride (hBN), with onsite energies
$\pm3.9$ eV, hopping $t=2.33$ eV and bond length 1.45 Å:

```python
import numpy as np
from pyqula import geometry
g = geometry.honeycomb_lattice() # honeycomb lattice, x is the armchair direction
h = g.get_hamiltonian(has_spin=True) # spinful, so both spins are counted
h.add_sublattice_imbalance(3.9/2.33) # onsite +-3.9 eV with t = 2.33 eV, as hBN

# sheet susceptibility chi_xxx in nm^2/V, bond length 1.45 A and t = 2.33 eV
(ws,chi) = h.get_shg_susceptibility(energies=np.linspace(0.,4.,401),
                                    nk=200,T=0.,delta=0.02,components="xxx",
                                    length_unit=1.45,energy_unit=2.33)
print("static chi_xxx in nm^2/V:",chi[0].real)
```

`chi` is an array with one value per frequency. With `components` a list,
for example `["xxx","xyy"]`, the result is a dictionary with one array per
component. The point group of hBN is $D_{3h}$. With $x$ along the
armchair direction, as in pyqula's honeycomb lattice, the only nonzero
components are $\chi_{xxx}=-\chi_{xyy}=-\chi_{yxy}=-\chi_{yyx}$, and
$\chi_{yyy}=0$.

![Real and imaginary part of the second-harmonic sheet susceptibility of a two-band hBN model against frequency, with half the gap and the gap marked.](img/optical_shg.png#only-light)
![Real and imaginary part of the second-harmonic sheet susceptibility of a two-band hBN model against frequency, with half the gap and the gap marked.](img/optical_shg-dark.png#only-dark)

The static value is $\chi_{xxx}=2.6\times10^{-3}$ nm$^2$/V. The gap is
$3.35t$, at the K point. The real part has its maximum,
$2.1\times10^{-2}$ nm$^2$/V, at $\hbar\omega=1.67t$, half the gap: this is
the $2\omega$ resonance with the band edge. The imaginary part has its
maximum, $2.3\times10^{-2}$ nm$^2$/V, at $\hbar\omega=1.95t$, half of the
transition at the M point, $3.90t$, where the density of states of the
transitions is largest. The $\omega$ resonances start at the gap, where
$\mathrm{Im}\,\chi_{xxx}$ changes sign, and they are weaker. The M transition
appears again at $\hbar\omega=3.90t$, as a small dip. Here `nk=100` already gives the curves to 1%, and
`nk=300` changes the curves by less than $2\times10^{-4}$ of their maximum
compared with `nk=200`.

??? example "Code of this figure (documentation/figures/optical_shg.py)"

    ```python
    --8<-- "optical_shg.py"
    ```

The keywords are

- `energies`: the fundamental frequencies $\hbar\omega$.
- `nk`: number of k-points per direction. As for the linear conductivity, a
  small `delta` needs a large `nk`.
- `T`: temperature of the Fermi function. A metal needs $T>0$.
- `delta`: the broadening $\eta$ of each frequency.
- `components`: one component such as `"xxx"`, or a list of them. Components
  along $z$ of a two-dimensional system are allowed, since the positions of
  the orbitals give the dipole along $z$.
- `length_unit`, `energy_unit` (only for `h.get_shg_susceptibility()`): the
  length unit of the geometry in Å and the energy unit of the Hamiltonian in
  eV, used to convert to SI units.
- `gauge`: `"velocity"` (the default) or `"length"`. The length gauge gives the
  same result for an insulator at $T=0$ with no degenerate bands, and is
  there as a check.

For a TMD monolayer, the three-band model of
[Transition metal dichalcogenides](hamiltonian.md#transition-metal-dichalcogenides)
gives for MoS$_2$ a static sheet susceptibility $|\chi_{yyy}|=0.59$
nm$^2$/V, with $x$ along the zigzag direction, of the same order as the
$0.3$ nm$^2$/V of a model with more orbitals (Trolle et al., PRB **89**,
235410 (2014)). With $x$ along zigzag the nonzero components are
$\chi_{yyy}=-\chi_{yxx}=-\chi_{xxy}=-\chi_{xyx}$.

The result counts the spin as the Hamiltonian does: a spinless Hamiltonian
gives the response of one spin. Excitonic effects are not included, and in
a TMD they move the resonances and change their weight.
`h.get_shg_conductivity()` returns $\sigma_{abc}$ in units of
$e^3 a^{3-d}/(\hbar t)$, with $a$ the length unit of the geometry and $d$
the dimension. The second harmonic works in one, two and three
dimensions, and in three dimensions it is a bulk quantity, as the linear
conductivity.

## Excitonic effects

We will now see how the attraction between the electron and the hole
changes the optical response of an insulator. In a two-dimensional
semiconductor this attraction binds excitons below the gap (see
[Excitons](excitons.md)), and both the absorption and the second harmonic
change. We solve the Bethe-Salpeter equation (BSE) at zero momentum in the
Tamm-Dancoff approximation, which gives the exciton energies $E_n$ and
their amplitudes $A^n_{cv}(\mathbf k)$ on the electron-hole pairs. The
dipole of exciton $n$ is

$$
R^a_{0n} = \sum_{cv\mathbf k} A^n_{cv}(\mathbf k)\, r^a_{vc}(\mathbf k),
$$

with $r^a_{vc}$ the interband dipole of the tight-binding model. The linear
susceptibility is a sum over excitons,

$$
\chi_{ab}(\omega) = \frac{q^2}{N_k V_{\rm cell}} \sum_n \left[
\frac{R^a_{0n} R^b_{n0}}{E_n-\omega} + \frac{R^b_{0n} R^a_{n0}}{E_n+\omega} \right],
\qquad \sigma_{ab} = -i\omega\chi_{ab},
$$

computed with `h.get_excitonic_optical_conductivity()`, so a bright
exciton is a peak in the absorption at $E_n$. The Hall part at zero
frequency, which is fixed by the Chern number, is taken from the bands, as
in [The Kubo formula](#the-kubo-formula).

The second harmonic also needs the dipole between two excitons,
$R_{nm}=\langle n|\mathbf r|m\rangle$. It contains the derivative of the
exciton amplitude with respect to $\mathbf k$, so the exciton has to be
smooth on the k-mesh. With it, the susceptibility is a sum over pairs of
excitons (Ruan, Chan and Louie, arXiv:2310.09674),

$$
\chi_{abc} = \frac{q^3}{2N_kV_{\rm cell}} \sum_{nm} \left[
\frac{R^a_{0n}R^b_{nm}R^c_{m0}}{(2\omega-E_n)(\omega-E_m)}
+ \frac{R^b_{0n}R^c_{nm}R^a_{m0}}{(2\omega+E_m)(\omega+E_n)}
- \frac{R^c_{0n}R^a_{nm}R^b_{m0}}{(\omega-E_m)(\omega+E_n)}\right]
+ (b\leftrightarrow c),
$$

with $\omega\to\omega+i\eta$ in every frequency. It is computed with
`h.get_excitonic_shg_conductivity()` and `h.get_excitonic_shg_susceptibility()`.
The first term shows the two resonances of the second harmonic: an exciton
at $2\hbar\omega=E_n$ and an exciton at $\hbar\omega=E_m$. Without the
interaction, `kernel="none"`, the excitons are free electron-hole pairs,
and both functions give back the results without excitons of the sections
above.

We take the gapped honeycomb lattice with the Coulomb tail of
[Excitons](excitons.md), which binds an exciton at $1.845t$, below the gap
$2t$:

```python
import numpy as np
from pyqula import geometry
from pyqula.bsetk.interaction import density_interaction
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False) # spinless, one spin species
h.add_sublattice_imbalance(1.0) # gapped honeycomb lattice, gap 2t at K
W = density_interaction(h,Vr=lambda r1,r2: 0.8/np.sqrt((r1-r2).dot(r1-r2)+0.25)) # Coulomb tail
zero = {(0,0,0):np.zeros(h.intra.shape,dtype=np.complex128)} # no interaction

kw = dict(energies=np.linspace(0.,2.6,261),nk=30,delta=0.04)
(ws,s) = h.get_excitonic_optical_conductivity(V=W,component="xx",**kw)
(ws,s0) = h.get_excitonic_optical_conductivity(V=zero,kernel="none",component="xx",**kw)
(ws,chi) = h.get_excitonic_shg_susceptibility(V=W,components="xxx",**kw)
(ws,chi0) = h.get_excitonic_shg_susceptibility(V=zero,kernel="none",components="xxx",**kw)
```

![Left: optical absorption of the gapped honeycomb lattice with and without the electron-hole interaction. Right: second-harmonic susceptibility with and without the interaction.](img/optical_excitonic.png#only-light)
![Left: optical absorption of the gapped honeycomb lattice with and without the electron-hole interaction. Right: second-harmonic susceptibility with and without the interaction.](img/optical_excitonic-dark.png#only-dark)

On the left, without the interaction the absorption starts at the gap.
With it, the exciton at $1.845t$ is a peak of $6.9\,e^2/h$, below the gap.
On the right, the exciton gives two peaks in the second harmonic. The
first is at $\hbar\omega=0.92t$, half the exciton energy, where
$|\chi_{xxx}|$ is $0.24$ nm$^2$/V against $0.055$ nm$^2$/V without the
interaction. The second is at $\hbar\omega=1.84t$, the exciton energy itself.
Even the static value grows, from $0.018$ to $0.028$ nm$^2$/V. Here the
default units are used, a length unit of 1 Å and an energy unit of 1 eV.
Above the gap both curves oscillate: these are the separate k-points of
the $30\times30$ mesh, which does not resolve the broadening
$\eta=0.04t$ there.

??? example "Code of this figure (documentation/figures/optical_excitonic.py)"

    ```python
    --8<-- "optical_excitonic.py"
    ```

The keywords are those of the sections above, `energies`, `nk`, `delta`,
`component` or `components`, and the ones of the BSE: the interaction `V`
(by default `h.V` from a mean field), `nv` and `nc` to keep a window of
bands, `kernel`, `screening` and `gauge`. A few things to keep in mind:

- The BSE is solved with all its excitons, so the cost grows as the cube of
  the number of electron-hole pairs, $N_k$ times the number of valence
  and conduction bands. A $30\times30$ mesh of a spinless two-band model
  (900 pairs) takes a few seconds. For spinful MoS$_2$ in the three-band
  model (eight pairs per k-point) it takes $3$ s on a $12\times12$ mesh,
  $23$ s on $18\times18$ and $2$ minutes on $24\times24$. A $30\times30$
  mesh needs about 3.3 GB, above the default limit of the BSE, which is
  raised with `max_memory=4`.
- For finer meshes use `method="kpm"`. Then the sums over excitons are
  done with a Chebyshev expansion of the BSE matrix at the complex energy
  $\hbar\omega+i\eta$, and the matrix is never diagonalized. The result
  is the same as with the default `method="dense"`, to $10^{-8}$ of its
  maximum. The cost grows only linearly with the number of pairs, and as
  $1/\eta$. It does not depend on the range of the interaction, and a
  screened interaction (`screening="rpa"`) works as well. For spinful
  MoS$_2$ with the interaction, 16 frequencies of one second-harmonic
  component take $0.3$ s on an $18\times18$ mesh and $3$ s on
  $60\times60$. A TMD tetralayer with every band takes less than a
  minute on $48\times48$.
- With `method="kpm"` the calculation also runs on a GPU. Call
  `gpu.set_gpu(True)` first (`from pyqula import gpu`). It runs in double
  precision. With `kpm_prec="single"` it is faster, and the result changes
  by up to $5\times10^{-5}$ of its maximum.
- Near a resonance the electron-hole pairs change over a momentum
  $\eta/v$, with $v$ the band velocity, and the second harmonic needs this
  change. So the broadening has to be at least $v\Delta k/2$, with
  $\Delta k$ the mesh spacing. For a three-band TMD tetralayer,
  $\eta=0.1$ eV on a $24\times24$ mesh is within 3% of the converged
  result without interaction, and $\eta=0.05$ eV needs a $48\times48$ mesh.
  The components that the symmetry forbids are zero on every mesh.
- Above the gap, keep all the conduction bands in `nc`. A narrow window
  misses transitions in the frequency range, and two close conduction
  bands at the edge of the window make the result jump from one k-point to
  the next.
- The excitons are in the Tamm-Dancoff approximation, which keeps only the
  absorption pairs. This is the usual choice and is accurate for bound
  excitons well below the gap.
- The band structure is used as it is. For a real material the gap
  should include the correction of a GW calculation. The simplest way is a
  scissor shift, `scissor=0.3`, which adds $0.3$ to the energy of every
  electron-hole pair and keeps the dipoles of the bands. So every exciton
  and every resonance moves up by $0.3$, and the binding energies do not
  change.

See `examples/2d/optical_conductivity/main.py`,
`examples/1d/optical_conductivity_chain/main.py` and
`examples/2d/second_harmonic_generation/main.py` and
`examples/2d/excitonic_optics/main.py` for runnable scripts (and
`examples/2d/shift_current/main.py` and `examples/2d/third_harmonic/main.py`
for the photocurrents and the third harmonic), and
`01_optical_conductivity.ipynb` and `02_drude_weight_f_sum_rule.ipynb` in
`jupyter-notebooks/functionalities/response_functions/` for executed
notebooks.
