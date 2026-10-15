# Altermagnets

## Spin splitting of an altermagnet

Let us now look at a magnet with no net magnetization whose two spin channels nevertheless
have different bands. An altermagnet is a collinear magnet whose spin-up and spin-down bands
are split even though it carries no net magnetization, and the natural measure of that order
is how far apart the two spin channels are pushed, and at which energy. Both quantities
below diagonalize the two spin blocks separately and pair the bands by index, so that every
k-point and band index $n$ has a splitting $\Delta_n(k) = E^\uparrow_n(k) - E^\downarrow_n(k)$
sitting at the mean energy $\bar E_n(k) = (E^\uparrow_n(k) + E^\downarrow_n(k))/2$.

The two differ in how they reduce that set. `h.get_spin_splitting_density()` broadens every
pair into a smooth weighted density, giving the typical splitting at each energy.
`h.get_spin_splitting_vs_energy()` instead keeps the largest $|\Delta|$ found in each energy
bin, so that its global maximum is a bound on the spin splitting anywhere in the Brillouin
zone. The distinction matters when reporting a single number for a material: a maximum taken
along one cut through reciprocal space (a circle of fixed radius, say) depends on choosing
where to look, while the binned maximum over a full mesh does not

```python
import numpy as np
from pyqula import specialhamiltonian

h = specialhamiltonian.square_altermagnet(am=1.)

# largest spin splitting in each energy bin, over the whole BZ
(es, ds) = h.get_spin_splitting_vs_energy(nk=100, nbins=400)
print("largest spin splitting anywhere in the BZ:", ds.max())

# the smooth counterpart, on the same axes
(xs, ys) = h.get_spin_splitting_density(nk=100, delta=1e-1, energies=es)
```

Both return `(energies, values)` with the same shape convention, so they can be plotted
together. Bins containing no states come back as `0.0` rather than `NaN`, and states falling
outside an explicitly requested energy window are dropped rather than piled onto the end
bins.

Two things about that index pairing decide whether the number you report is the number you
meant.

The first is reassuring. In a collinear altermagnet the two spin channels are related by a
point-group operation acting on $k$ rather than by a state-by-state correspondence at fixed
$k$: the sorted spectra satisfy $E^\uparrow_n(k) = E^\downarrow_n(Rk)$ exactly, so that
$\Delta_n(k) = E^\downarrow_n(Rk) - E^\downarrow_n(k)$ compares the same sorted index within
one channel at two related momenta. No band-identification ambiguity survives, and the usual
worry, that a splitting exceeding the band spacing makes the $n$-th up and $n$-th down band
different bands, does not arise, because the symmetry supplies the correspondence. For a new
system it pays to identify $R$: compare `sorted(E_up(k))` with `sorted(E_dn(Rk))` over the
point group and look for the operation giving `~1e-14`. For the square altermagnet it is the
`k1<->k2` mirror.

The second is a real dependence that symmetry does not remove. The band index $n$ labels
whatever set of bands your unit cell produces, and folding changes that set: on a supercell,
bands at one $k$ come from several primitive k-points, and pairing them by index compares
across those. The reported maximum then comes out too small. Give this the true magnetic unit
cell: if the converged order repeats with a shorter period than the cell it was solved in,
the result is a lower bound rather than the maximum.

Both require spin to be a good quantum number, since they keep one spin block and discard the
off-diagonal one. `h.get_spin_splitting_vs_energy()` checks this and raises if the spin
off-diagonal block of the Bloch Hamiltonian is not negligible: with Rashba coupling, any
other spin-orbit term, or non-collinear magnetic order the splitting defined above is not a
meaningful quantity, and a silently wrong number would be worse than an error.
Diagonalization is dense throughout, deliberately: a sparse solver returns only the
eigenvalues nearest $E=0$, and the splitting commonly peaks far away from there.

See `examples/2d/spin_splitting_vs_energy/main.py` for a runnable version plotting both curves (and showing the redundant-cell trap), `examples/2d/altermagnetism_density/main.py` for the density alone, and `jupyter-notebooks/functionalities/single_particle_hamiltonians/11_altermagnet_spin_splitting.ipynb` for an executed notebook.



## Nonlinear spin current as a measurement of altermagnetic order

The spin splitting of the previous chapter tells you how big the altermagnetic order is. A
harder question is which kind it is, d-wave, g-wave or i-wave, and that question turns out to
have a purely electrical answer, one that needs no spin-orbit coupling at all. We will now see
how to read the wave index off the nonlinear spin current, with
`h.get_nonlinear_drude_orders()`.

The idea is the following. The spin-splitting form factor of an X-wave collinear magnet is a
harmonic of order `l+1` in reciprocal space: `kx*ky` for d-wave, `kx*ky*(kx^2-ky^2)` for
g-wave, `kx*ky*(3kx^2-ky^2)*(kx^2-3ky^2)` for i-wave. In the semiclassical Boltzmann
treatment, the `l`-th order nonlinear Drude conductivity is a Brillouin-zone integral of the
`(l+1)`-th derivative of the band energy,

```
sigma_s^{x^l1 y^l2 ; b} = (-e/hbar)^(l+1)/(i omega + 1/tau)^l
        * Int d^Dk/(2 pi)^D f_s^(0) d^(l+1) eps_s/(dk_x^l1 dk_y^l2 dk_b)
```

with `sigma_spin = (sigma_up - sigma_dn)/2` and `sigma_charge = sigma_up + sigma_dn`. A
harmonic of order `l+1` has a nonzero constant `(l+1)`-th derivative, and nothing below it
survives the zone integral, so the spin response switches on at exactly one order and is
silent below it:

| wave | d | f | g | i |
|---|---|---|---|---|
| lowest order `l` | 1 | 2 | 3 | 5 |

The p-wave magnet, whose form factor is `kx`, would sit at `l = 0` by this counting, and it is
not in the table for two reasons. First, the zeroth order vanishes for every band of every
Hamiltonian: with $F'=f$ the integrand $f\,\partial\epsilon/\partial k_b=\partial F(\epsilon)/\partial k_b$
is a total derivative, whose zone integral is zero, meaning that there is no current in
equilibrium. Second, on the lattice the p-wave splitting $J\sin k_x$ combines with the hopping
$2t\cos k_x$ into a single cosine of amplitude $\sqrt{4t^2+J^2}$, so the two spin bands are the
same band shifted rigidly along $k_x$ by opposite amounts, and a zone integral does not see
that shift. A p-wave magnet has spin-split Fermi surfaces and no spin current at any order.
Note that on a finite k-mesh these two zeros come out as small numbers, since the shift is not
a translation of the mesh: at `nk=48` the p-wave magnet gives $3\times10^{-4}$ at the even
orders, and this falls exponentially with `nk`, to $10^{-11}$ at `nk=384`. The zeros below
the threshold of the d-, f-, g- and i-wave magnets are forced by symmetry and are exact on
any mesh, so an entry that shrinks as the mesh is refined is a zero and not a threshold.

Reading off the lowest order at which a nonlinear spin current appears therefore identifies
the wave index. This is the content of Ezawa, *Phys. Rev. B* **111**, 125420
([arXiv:2411.16036](https://arxiv.org/abs/2411.16036)). For an i-wave altermagnet we write

```python
from pyqula import specialhamiltonian

h = specialhamiltonian.iwave_altermagnet(J=0.3)   # or dwave_/fwave_/gwave_

# largest |sigma_spin| over every component of each order l
orders = h.get_nonlinear_drude_orders(lmax=6, nk=48, T=0.02, mu=-5.5)
# -> zero to machine precision for l = 0..4, nonzero at l = 5

# one component, and every component of a given order
s = h.get_nonlinear_drude_conductivity(field="yyyyy", current="x", mu=-5.5)
c = h.get_nonlinear_drude_components(5, nk=48, mu=-5.5)   # {"x^l1 y^l2;b": value}
```

What comes back from `h.get_nonlinear_drude_orders()` is, for each order `l` up to `lmax`,
the largest spin conductivity over every component of that order: zero to machine precision
for `l` from 0 to 4 and nonzero at `l = 5`, which is the i-wave fingerprint. `field` carries
one character per power of the electric field, so `field="yyyyy"` is the fifth-order response
to $E_y$; `current` is the direction the current is measured in, and `channel` picks
`"spin"`, `"charge"`, `"up"` or `"dn"`. Units are $e=\hbar=1$, with the zone integral
normalized as a density, the same convention `fermi_volume` uses.

The X-wave models themselves are `specialhamiltonian.xwave_magnet(wave=...)`, with
`pwave_magnet`, `dwave_altermagnet`, `fwave_magnet`, `gwave_altermagnet` and
`iwave_altermagnet` as named aliases. These are Ezawa's models, each on the lattice its
symmetry requires: square for p, d and g, triangular for f and i.

Before quoting a result, there are a few things to keep in mind. First, the orders above the
threshold do not vanish: on a lattice the form factor carries higher harmonics beyond the
leading one, so a d-wave altermagnet responds at `l = 1` but also at `l = 3` and `l = 5`, and
it is the absence of everything below the threshold that carries the information, not the
presence of a single isolated order. Second, an insulator gives zero at every order, which is
a trap: with the chemical potential in a gap every valence band is full and every conduction
band empty, the zone integral vanishes identically at every order including the fifth, and a
gapped system reproduces the "nothing below fifth order" pattern trivially. The fingerprint
only identifies i-wave order if the fifth order is also shown to be present, and that needs a
metal. Third, a spin-degenerate state is flagged rather than answered: the spin channel only
means something if the two spin channels have different bands, and a compensated Neel state
on a bipartite lattice is $PT$ symmetric and exactly spin degenerate. That is an
antiferromagnet rather than an altermagnet, and what comes back there is noise that does not
look small, so a warning is raised whenever the splitting falls below $10^{-10}$ of the
bandwidth; a real altermagnet stays far above that floor.

Reality is itself a selection rule, and a useful one when building a model. With real
hoppings and no spin-orbit coupling $\epsilon_s(k)=\epsilon_s(-k)$, so the splitting is even
under $\phi\to\phi+180^\circ$ and every odd harmonic is forbidden. A C3 axis then removes
everything not divisible by three, leaving only the harmonics of order $6, 12, \dots$, that
is, $l=5, 11, \dots$, meaning that a real, spin-orbit-free, C3-symmetric cell is necessarily
i-wave, and no vacancy pattern or supercell can make it f-wave. The models here bear this out:
p-wave and f-wave need imaginary hoppings, while d-, g- and i-wave are real.

Finally, spin must be a good quantum number. The formula treats each spin channel on its own,
so a Hamiltonian with spin-orbit coupling is rejected rather than mistreated: with Rashba or
Kane-Mele terms the nonlinear response picks up quantum-metric and Berry-curvature-dipole
contributions ([arXiv:2409.09241](https://arxiv.org/abs/2409.09241)) that this formula does
not contain.

Multiorbital cells (superlattices, antidot lattices, multilayers) are supported, and are the
interesting case. The band derivatives there are taken analytically, including at the
degeneracies that a C3 or C6 axis forces at high-symmetry points, so an order the selection
rule forbids comes out as an exact zero rather than as small numerical noise.

See `examples/2d/xwave_nonlinear_spin_current/main.py` for a runnable version printing the
whole selection-rule table, and
`jupyter-notebooks/functionalities/single_particle_hamiltonians/12_nonlinear_spin_conductivity.ipynb`
for an executed notebook.
