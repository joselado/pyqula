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

The result counts the spin as the Hamiltonian does: a spinless Hamiltonian
gives the response of one spin. Excitonic effects are not included, and in
a TMD they move the resonances and change their weight.
`h.get_shg_conductivity()` returns $\sigma_{abc}$ in units of
$e^3 a^{3-d}/(\hbar t)$, with $a$ the length unit of the geometry and $d$
the dimension.

See `examples/2d/optical_conductivity/main.py`,
`examples/1d/optical_conductivity_chain/main.py` and
`examples/2d/second_harmonic_generation/main.py` for runnable scripts, and
`01_optical_conductivity.ipynb` and `02_drude_weight_f_sum_rule.ipynb` in
`jupyter-notebooks/functionalities/response_functions/` for executed
notebooks.
