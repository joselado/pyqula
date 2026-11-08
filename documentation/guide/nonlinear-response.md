# Nonlinear response

At second order in the electric field, a crystal gives a current that goes
as $E^2$. It needs broken inversion symmetry, because under inversion the
current and the field both change sign. In a metal under a dc field this is
nonlinear transport, and the nonlinear Hall effect is one example of it.
Under light it is the bulk photovoltaic effect, a dc current from
illumination alone, and the second harmonic of
[Optical conductivity](optical-conductivity.md#second-harmonic-generation).
At third order the current goes as $E^3$, inversion is not needed, and
graphene has a third harmonic. In this page we see these responses one by
one: the dc response of a metal in powers of the relaxation time, the Berry
curvature dipole, the photocurrents, and the response at any order and any
set of frequencies.

All of them use the conventions of the optical conductivity: $\hbar=e=1$,
energies in units of the hopping $t$, lengths in units of the lattice
constant $a$, the carriers are electrons, and the Fermi energy is at zero.
The $n$-th order conductivity is per unit cell area in 2D, in units of
$e^{n+1}a^{n+1-d}/(\hbar t^{n-1})$ in $d$ dimensions. The first index of a
component is the current and the others are the fields, so `"xyy"` is the
current along $x$ driven by two fields along $y$.

## Nonlinear transport of a metal

We will now see the second-order dc current of a metal,

$$
J_a = \sigma_{abc} E_b E_c ,
$$

with $\sigma_{abc}$ symmetric in $b$ and $c$. Let us start from the
Boltzmann equation with a relaxation time $\tau$. The field shifts the Fermi
sea by $\tau E$ in momentum, once at first order and twice at second order.
The current comes from the band velocity and from the anomalous velocity of
the Berry curvature. Together with the field correction of the Bloch states
this gives three terms, one for each power of $\tau$,

$$
\begin{aligned}
\sigma^{\rm Drude}_{abc} &= -\tau^2 \left\langle w\, u_a u_{bc} \right\rangle , \\
\sigma^{\rm BCD}_{abc} &= \frac{\tau}{2} \left\langle w \left(u_c\Omega_{ab} + u_b\Omega_{ac}\right) \right\rangle , \\
\sigma^{\rm QM}_{abc} &= -\left\langle w \left[u_a \tilde G_{bc} - \tfrac12 \left(u_b\tilde G_{ca} + u_c\tilde G_{ab}\right)\right] \right\rangle
 - \frac12 \left\langle w\, \partial_a G_{bc} \right\rangle .
\end{aligned}
$$

Here $\langle X\rangle = \frac{1}{N_k V_{\rm cell}}\sum_{\mathbf k,n} X_n$ is
the sum over the bands $n$ and the k-mesh, $w_n = -\partial f/\partial E$ at
$E_n$ is the weight of the Fermi surface, $u_a$ is the band velocity,
$u_{ab}$ the inverse mass, $\Omega_{ab}$ the Berry curvature, $G_{ab}$ the
quantum metric, and
$\tilde G_{ab} = 2\,\mathrm{Re}\sum_{m\neq n} v^a_{nm}v^b_{mn}/(E_n-E_m)^3$
the band-normalized quantum metric, also called the Berry connection
polarizability. The first term is the nonlinear Drude term, the curvature
of the bands seen by the shifted Fermi sea. The second is the Berry curvature
dipole term, the anomalous velocity of the shifted Fermi sea. The third is
the quantum-metric term. It does not depend on $\tau$: it comes from the
field correction of the Bloch states and not from the shift of the Fermi
sea.

Every term is a sum over the Fermi surface, so a band insulator gives zero.
The symmetries tell the terms apart. Time reversal makes $E_n(\mathbf k) =
E_n(-\mathbf k)$, and this kills the Drude and the quantum-metric terms. The
product $PT$ of inversion and time reversal makes the Berry curvature zero at
every $\mathbf k$, and this kills the Berry curvature dipole term. So a
time-reversal symmetric metal has only the dipole term, and a $PT$-symmetric
antiferromagnet has only the Drude and the quantum-metric terms.

In an experiment one measures the voltage at $2\omega$ under a
low-frequency current at $\omega$, as a function of temperature. The
linear conductivity $\sigma_{xx}$ goes as $\tau$, so the second-order signal
plotted against $\sigma_{xx}$ is a parabola: the quadratic part is the
Drude term, the linear part the dipole term, and the constant part the
quantum-metric term.

We take the doped gapped honeycomb lattice under uniaxial strain, with a
Haldane term. The staggered onsite energy breaks inversion, the Haldane term
breaks time reversal and the strain breaks the $C_3$ rotation, so the three
terms are allowed. `h.get_nonlinear_dc_conductivity()` computes them:

```python
import numpy as np
from pyqula import geometry, strain
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False) # spinless, one spin species
h.add_sublattice_imbalance(0.5) # staggered onsite energy, gap 1.0
strain.uniaxial_strain(h,d=np.array([1.,0.4,0.]),s=0.15) # breaks C3
h.shift_fermi(-0.9) # chemical potential at mu = 0.9, in the conduction band
h.add_haldane(0.1) # breaks time reversal

t = h.get_nonlinear_dc_conductivity(nk=150,T=0.05,tau=1.,components=["yxx","xyy"])
for term in ["drude","bcd","qmd","total"]: # tau^2, tau^1, tau^0 and the sum
    print(term,t[term]["yxx"],t[term]["xyy"])
```

`t` is a dictionary with one entry per term, `"drude"`, `"bcd"` and `"qmd"`,
each with its power of $\tau$ included, and their sum `"total"`. Each entry
is a dictionary with one number per component. For $\sigma_{yxx}$, the
transverse current under a field along $x$, the Drude term is
$-4.18\times10^{-2}$, the dipole term $2.63\times10^{-3}$ and the
quantum-metric term $8.34\times10^{-3}$, in units of $e^3a/(\hbar t)$ with
$\tau=1$ in units of $\hbar/t$. With `tau=2.` the Drude term becomes four
times larger, the dipole term two times, and the quantum-metric term does
not change. Doubling the mesh to `nk=300` changes none of the numbers in
the third digit. With the Haldane term removed, the Drude and the
quantum-metric terms are zero to $10^{-16}$, as time reversal requires.

Every term has the weight $w_n$, which is nonzero only within a few $T$ of
the Fermi energy, so the k-mesh has to resolve the Fermi surface on the
scale of $T$: the energy step between two neighboring k-points has to be
smaller than $T$. The way to
check this is to double `nk` at fixed `T` and see that nothing changes.

The quantum-metric term is not unique. The formulas in use differ by terms
of order $\tau^0$ that depend on how the scattering is modeled. The default,
`qmd="shibata"`, is the one written above. `qmd="gao"` keeps only the first
part, which is transverse ($\sigma_{aaa}=0$), and `qmd="kaplan"` replaces
the second part by $-\langle w\,u_a\tilde G_{bc}\rangle$. The single parts are
available as `term="qmd_hall"`, `term="qmd_disp"` and `term="qmd_intra"`.

At third order, `order=3` gives the current $J_a = \sigma_{abcd}E_bE_cE_d$
with two terms: the nonlinear Drude term $\tau^3\langle f\,\partial_a
\partial_b\partial_c\partial_d E_n\rangle$ and the term linear in $\tau$ of the
Berry connection polarizability $\tilde G$. The components have four
labels, such as `"xxyy"`. The third-order terms with $\tau^2$ and $\tau^0$
are not included.

The keywords are

- `term`: `"all"` (the default, a dictionary), or one term.
- `tau`: the relaxation time $\tau$, in units of $\hbar/t$.
- `qmd`: `"shibata"` (the default), `"gao"` or `"kaplan"`, the convention of
  the quantum-metric term.
- `order`: 2 (the default) or 3.
- `channel`: `"charge"` (the default), or `"spin"`, `"up"`, `"dn"` for a
  Hamiltonian that conserves spin.
- `nk`: number of k-points per direction.
- `T`: temperature, must be positive.
- `components`: one component such as `"xyy"`, a list of them, or all of
  them.

Degenerate bands, as in a $PT$-symmetric antiferromagnet where every band is
doubly degenerate, are handled: the result does not depend on how the
states inside a degenerate pair are chosen. The 2D, 1D and 3D cases are all
covered, except for the third-order Drude term, which is 1D and 2D only. A
runnable script is in `examples/2d/nonlinear_hall/main.py`. The X-wave index
of an altermagnet from the nonlinear spin current is in
[Altermagnets](altermagnets.md).

## The Berry curvature dipole

We will now see the quantity behind the nonlinear Hall effect of a metal
with time reversal. When the Fermi sea is shifted by the field, the Berry
curvature of the occupied states does not average to zero any more. What
measures this is the dipole of the Berry curvature on the Fermi surface,

$$
D_a = \frac{1}{N_k V_{\rm cell}} \sum_{\mathbf k,n}
\left(-\frac{\partial f}{\partial E}\right)_{E_n} u^n_a\,\Omega^n_{xy}
= \frac{1}{N_k V_{\rm cell}} \sum_{\mathbf k,n} f(E_n)\,\partial_a\Omega^n_{xy} ,
$$

computed with `h.get_berry_curvature_dipole()`. In 2D it gives the dipole
term of the section above, $\sigma_{xyy}=\tau D_y$ and $\sigma_{yxx}=-\tau
D_x$, meaning that a field along $y$ gives a transverse current along $x$ of
strength $\tau D_y$. With time reversal the Berry curvature is odd in
$\mathbf k$, so the total is zero, but its dipole is even and allowed. The
dipole is an in-plane vector, so it vanishes under a $C_3$ rotation, and a
crystal has a dipole only with at most one mirror line. This is why the
nonlinear Hall effect is measured in low-symmetry crystals, such as bilayer
WTe$_2$, which has a single mirror line.

We take the doped gapped honeycomb lattice, with and without strain:

```python
import numpy as np
from pyqula import geometry, strain
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False) # spinless, one spin species
h.add_sublattice_imbalance(0.5) # gapped honeycomb, time reversal kept
h.shift_fermi(-1.1) # chemical potential at mu = 1.1
D0 = h.get_berry_curvature_dipole(nk=150,T=0.05) # with C3
strain.uniaxial_strain(h,d=np.array([1.,0.4,0.]),s=0.15) # breaks C3
D = h.get_berry_curvature_dipole(nk=150,T=0.05) # without C3
print(D0,D)
```

`D0` and `D` are arrays $[D_x, D_y]$. Without strain the dipole is zero to
$10^{-17}$. With the strain it is $D=(-1.00,-0.84)\times10^{-2}$ in units of
the lattice constant. Scanning the chemical potential from the conduction
band edge at $0.5t$ up to $2.5t$, $|D_x|$ is largest at $\mu=1.1t$, and
both components keep their sign. `nk=300` gives the same numbers to eight
digits. The keywords are `nk`, `T`, `channel` and `degeneracy_tol`, as in
`h.get_nonlinear_dc_conductivity()`. In 3D the result is the tensor
$D_{ab}$, with the Berry curvature as a pseudovector
$\Omega_z=\Omega_{xy}$ and so on, and it is dimensionless.

## Photocurrents

We will now see the dc current that light gives in a crystal without
inversion, the bulk photovoltaic effect. For a field
$E(t)=\mathrm{Re}\,E_0e^{-i\omega t}$ the rectified current is

$$
J_a = \frac12\,\sigma_{abc}(0;\omega,-\omega)\,E_{0b}E^*_{0c} ,
$$

and it has two parts with a different physical origin. In the first part,
light excites an electron from band $n$ to band $m$, and the two bands have
different velocities. Every absorbed photon adds the velocity difference to
the current, so the current grows in time until the relaxation stops it.
This is the injection current, with a rate

$$
\frac{dJ_a}{dt} = \eta_{abc}\,E_{0b}E^*_{0c} ,
\qquad
\eta_{abc} = \frac{\pi q^3}{2}\,\frac{1}{N_kV_{\rm cell}}\sum_{\mathbf k,n\neq m}
(f_n-f_m)\,(v^a_{mm}-v^a_{nn})\,r^b_{mn}r^c_{nm}\,
\delta(\hbar\omega - E_m + E_n) ,
$$

where $r^a_{nm} = v^a_{nm}/(i(E_n-E_m))$ is the interband Berry connection
and $q=-1$ the charge. In the calculation the delta function is a Lorentzian
of width `delta`. In the second part, the excited electron is created at
a different position in the unit cell than the hole it leaves. Each
transition moves charge by a fixed shift vector, and this gives a steady
current. This is the shift current, $\sigma^{\rm shift}_{abc}(\omega)$, with
$J_a = \frac12\sigma^{\rm shift}_{abc}E_{0b}E^*_{0c}$.

In the calculation every frequency carries the broadening $\delta$ (the
$\eta$ of [Optical conductivity](optical-conductivity.md), renamed here
because $\eta$ is the injection rate), and the injection rate shows up as a
pole: the rectified conductivity is
$\eta_{abc}/\delta + \sigma^{\rm shift}_{abc}$. The shift current is what is
left after the pole is removed. `h.get_injection_current()` computes
$\eta_{abc}(\omega)$ and `h.get_shift_current()` computes
$\sigma^{\rm shift}_{abc}(\omega)$.

The symmetries again tell the two parts apart. $\eta$ is complex, with
$\eta_{acb}=\eta^*_{abc}$. Its real part is the injection under linearly
polarized light, and it vanishes with time reversal. Its imaginary part is
the injection under circularly polarized light, the circular photogalvanic
effect, and it is allowed with time reversal. For light with
$E_0=(1,i,0)/\sqrt2$ the rate is
$dJ_a/dt = (\mathrm{Re}\,\eta_{axx}+\mathrm{Re}\,\eta_{ayy})/2 +
\mathrm{Im}\,\eta_{axy}$. The shift current needs only broken inversion. So
in a nonmagnetic insulator under linearly polarized light the shift current
is the whole response.

We take the gapped honeycomb lattice, which has time reversal, for the shift
current, and add a Haldane term for the injection current:

```python
import numpy as np
from pyqula import geometry
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False) # spinless, one spin species
h.add_sublattice_imbalance(0.5) # gapped honeycomb, gap 1.0

ws = np.linspace(0.,4.,201) # frequencies
(ws,s) = h.get_shift_current(energies=ws,nk=100,T=0.,delta=0.05,
                             components=["xxx","xyy","yyy"])

h2 = g.get_hamiltonian(has_spin=False)
h2.add_haldane(0.2) # breaks time reversal
h2.add_sublattice_imbalance(0.3) # gap 1.2
(ws,eta) = h2.get_injection_current(energies=ws,nk=100,T=0.,delta=0.05,
                                    components=["yyy","yxx","xxx"])
```

`s` and `eta` are dictionaries with one complex array per component, one
value per frequency. The shift current starts at the gap, $1.0t$, and has
its maximum, $\sigma_{xxx}=8.7\times10^{-2}\,e^3a/(\hbar t)$, at
$\hbar\omega=1.1t$, just above the gap. Below the gap there is only the tail
of the broadening, which halves when `delta` halves. The point group of the
lattice is $D_{3h}$, so $\sigma_{xyy}=-\sigma_{xxx}$ and $\sigma_{yyy}=0$
exactly, with $x$ along the armchair direction. The injection rate of the
Haldane model is largest at $\hbar\omega=1.76t$, with
$\eta_{yyy}=1.2\times10^{-2}$ in units of $e^3a/\hbar^2$, and
$\eta_{yxx}=-\eta_{yyy}$, $\eta_{xxx}=0$. Without the Haldane term the
injection rate is zero to $10^{-16}$. `nk=200` changes the maximum of the
shift current by less than $0.1\%$ and the injection rate by $1\%$.

The keywords are those of `h.get_shg_conductivity()`: `energies`, `nk`, `T`
(equal to `delta` by default), `delta` and `components`. A small `delta`
needs a large `nk`, as for the absorption, and with broken time reversal the
pole $\eta/\delta$ makes the rectified conductivity converge more slowly in
`nk` than the second harmonic. In a metal the rectified conductivity also
has terms from the Fermi surface, such as the Berry curvature dipole, and
then the part left after the pole is not only the shift current. A
runnable script is in `examples/2d/shift_current/main.py`.

## Any order and any frequency

We will now see the general nonlinear conductivity, the current at the sum
frequency $\omega_1+\dots+\omega_n$ driven by $n$ fields,

$$
J_b(\omega_1+\dots+\omega_n) = \sigma_{ba_1\dots a_n}(\omega_1,\dots,\omega_n)\,
E_{a_1}(\omega_1)\cdots E_{a_n}(\omega_n) ,
$$

computed with `h.get_nonlinear_conductivity()` at any order $n$. It solves
the equation of motion of the density matrix order by order in the field,
in the velocity gauge, with the derivatives of $H(\mathbf k)$ up to order
$n+1$ as the vertices. This is the method of the second harmonic in
[Optical conductivity](optical-conductivity.md#second-harmonic-generation),
for any number of fields, and it is exact for the tight-binding model with
any number of bands. The field frequencies are $\omega_j = c_j\omega$, with
the multipliers $c_j$ given by `frequencies`:

- `frequencies=None`: all $c_j=1$, the $n$-th harmonic.
- `frequencies=(1,-1)`: the rectified current under light, the total of the
  section above.
- `frequencies=(1,2)`: the sum frequency $\omega+2\omega$.
- `frequencies=(1,1,0)`: the second harmonic induced by a static field.

For a real field $E(t)=\mathrm{Re}\,E_0e^{-i\omega t}$ the $n$-th harmonic of
the current is $2^{1-n}\,\mathrm{Re}[\sigma(\omega,\dots,\omega)E_0^n
e^{-in\omega t}]$, so $\frac12$ at second order and $\frac14$ at third.
`h.get_nonlinear_susceptibility()` returns
$\chi^{(n)} = \sigma^{(n)}/(-i\Omega\,\epsilon_0)$ in SI units, with $\Omega$
the sum frequency, in nm$^{n+2-d}$/V$^{n-1}$.

At third order inversion is not needed, so graphene has a third harmonic. We
take doped graphene:

```python
import numpy as np
from pyqula import geometry
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False) # spinless, one spin species
h.shift_fermi(-0.5) # chemical potential at mu = 0.5

comps = ["xxxx","xxyy","xyxy","xyyx","yyyy"]
(ws,s) = h.get_nonlinear_conductivity(order=3,energies=np.linspace(0.1,3.,59),
                                      nk=100,T=0.05,delta=0.1,components=comps)
print(s["xxyy"]/s["xxxx"]) # 1/3
```

`s` is a dictionary with one complex array per component. The point group of
graphene, $D_{6h}$, leaves one independent in-plane number:
$\sigma_{xxxx}=\sigma_{yyyy}=\sigma_{xxyy}+\sigma_{xyxy}+\sigma_{xyyx}$. The
three fields are at the same frequency, so the last three components are
equal, and each one is $\sigma_{xxxx}/3$. The code gives this ratio to
$10^{-15}$ on any mesh. $|\sigma_{xxxx}|$ is $40$ in units of
$e^4a^2/(\hbar t^2)$ at $\hbar\omega=0.1t$, where the Fermi surface dominates,
and falls by four orders of magnitude up to $\hbar\omega=3t$. `nk=200`
changes it by less than $0.3\%$. A runnable script is in
`examples/2d/third_harmonic/main.py`.

The dc limit needs care. `energies=[0.]` gives the static response with the
same broadening $\delta$ on every field, which is not the relaxation-time
result of [Nonlinear transport of a metal](#nonlinear-transport-of-a-metal).
The two cannot be matched with $\delta=1/\tau$: at second order the Drude
term goes as $1/(2\delta^2)$ where the relaxation time gives $\tau^2$, the
dipole term as $1/\delta$, and the term of order $\delta^0$ agrees with none
of the quantum-metric conventions. For the dc transport of a metal use
`h.get_nonlinear_dc_conductivity()`. For an insulator the static limit is
finite, and `h.get_nonlinear_susceptibility()` at `energies=[0.]` with a
small `delta` gives the static susceptibility.

The keywords are

- `order`: the order $n$ of the response.
- `energies`: the frequencies $\hbar\omega$.
- `frequencies`: the multipliers $c_j$, one per field.
- `nk`, `T`, `delta`: as for the second harmonic. A metal needs $T>0$ and a
  mesh that resolves both $T$ and `delta`.
- `components`: strings of $n+1$ labels, the current first.
- `prec`: `"double"`, or `"single"` on the GPU.

With `gpu.set_gpu(True)` (`from pyqula import gpu`) the calculation runs on
the GPU. `prec="single"` is faster still, and it is accurate only when every
field frequency is larger than `delta`, because the static parts of the
response cancel to many digits. See [GPU](gpu.md).
