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

The formula works for one- and two-dimensional Hamiltonians.
Superconducting (Nambu) Hamiltonians and three-dimensional ones raise
`NotImplementedError`; for a Nambu Hamiltonian the hole block carries the
opposite charge, so the current is not $i[H,r]$.

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
the dimension.

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
- The band structure is used as it is. For a real material it should
  include the gap correction of a GW calculation, or a scissor shift.

See `examples/2d/optical_conductivity/main.py`,
`examples/1d/optical_conductivity_chain/main.py` and
`examples/2d/second_harmonic_generation/main.py` and
`examples/2d/excitonic_optics/main.py` for runnable scripts, and
`01_optical_conductivity.ipynb` and `02_drude_weight_f_sum_rule.ipynb` in
`jupyter-notebooks/functionalities/response_functions/` for executed
notebooks.
