# Chebyshev kernel polynomial (KPM) methods

Let us now see what to do when a system is too large to diagonalize. Exact
diagonalization builds the full spectrum, and for a system with millions of
sites that is neither possible nor needed: an observable such as the density of
states is a trace, and a trace can be estimated without ever knowing the
eigenstates. This is the idea of the kernel polynomial method, which expands
the density of states in Chebyshev polynomials of the Hamiltonian, obtains the
expansion coefficients (the moments) by repeatedly multiplying a random vector
by the sparse Hamiltonian, and averages over a few random vectors, a stochastic
estimate of the trace. Passing `mode="KPM"` to `h.get_dos()`, the mode listed
in the density of states section, switches to this expansion, which never
builds the full spectrum and scales to systems with millions of sites on a
single core. A sparse Hamiltonian (`is_sparse=True`) is the natural setting,
since the method only multiplies vectors by the Hamiltonian and never needs
the dense matrix, although a dense one is accepted too

```python
from pyqula import geometry
import numpy as np
g = geometry.chain()
g = g.get_supercell(3000) # a big supercell
g.dimensionality = 0
h = g.get_hamiltonian(is_sparse=True,has_spin=False)
(x,y) = h.get_dos(mode="KPM",
            energies=np.linspace(-3.0,3.0,200), # energies
            delta=1e-2, # energy resolution, half width of the peak of a level
            ntries=10 # number of random vectors for the stochastic trace
            )
```

`x` is the energy grid and `y` the density of states on it, the same quantity
that exact diagonalization would give for a chain of this length. Two numbers
set what you get. `delta` is the energy resolution of the expansion, the half
width at half maximum of the peak that a single level gives, which is the width
it gives the Lorentzian of `mode="ED"`, so that the two modes can be laid on top
of each other. The expansion takes about $1.85\,{\rm scale}/\delta$ polynomials
for it, so a sharper density of states costs proportionally more matrix-vector
products, and each peak comes out nearly a Gaussian rather than a Lorentzian,
about one and a half times taller and without the long tails. `delta` also has
to stay larger than the spacing between the levels, $2\pi t/N\approx 0.002$ in
the middle of the band of this chain, or `y` shows the individual levels.
`ntries` is the number of random vectors in the stochastic trace, and the noise
in `y` goes down as the inverse square root of their number.
The expansion also needs the whole spectrum inside $[-{\rm scale},{\rm scale}]$,
since a Chebyshev polynomial grows exponentially outside $[-1,1]$, and the
default is `scale=10` in units of the hopping; a Hamiltonian whose spectrum
reaches further, through a large onsite energy or exchange field, needs a
larger `scale`, and one that is too small raises a `ValueError` rather than
returning a density of states that means nothing. The same holds for the
routines of the `kpm` module below.

The same expansion also gives non-local correlators and Green's functions
without inverting a matrix, through the lower-level `kpm` module

```python
from pyqula import kpm
(x,y) = kpm.dm_ij_energy(h.intra,npol=200,i=0,j=9,ne=1000)
```

`kpm.dm_ij_energy()` returns the correlator between sites $i$ and $j$ resolved
in energy, $\pi\sum_n \psi_n(i)\psi_n^*(j)\,\delta(E-E_n)$, the spectral part of
the Green's function $G_{ij}(E)$: for $i=j$ it is $\pi$ times the local density
of states at that site, and for two different sites it integrates to zero over
the energy, since the eigenstates form a complete basis, and its sign along the
energy axis tells in which part of the spectrum the two sites are correlated.
`npol` is the number of polynomials $n_{\rm pol}$ and sets the resolution, a
peak of half width $1.85\,{\rm scale}/n_{\rm pol}$ for each level, the inverse
of what `delta` did above. The Green's function itself, real part included,
comes from the same moments with `kpm.correlator0d()`, which returns the
energies, ${\rm Re}\,G_{ij}$ and $-{\rm Im}\,G_{ij}$, and with
`kernel="lorentz"` broadens each pole into a Lorentzian of half width
$1.5\,{\rm scale}/n_{\rm pol}$, the $G_{ij}(E+i\delta)$ that a matrix inversion
gives.

See `examples/0d/kpm_dos/main.py` and `examples/0d/kpm_correlator/main.py`
for runnable versions, including a comparison of the KPM correlator against
the exact Green's function calculation, and the notebooks in
`jupyter-notebooks/functionalities/chebyshev_kpm_algorithms/` for the local
and non-local spectral functions and for a system too large to diagonalize.

The Chebyshev moments can be computed on a GPU instead, by setting the
package-wide switch of "Running on a GPU" before calling any of the KPM
functions, and `kpm_prec` chooses the precision of the moments:

```python
from pyqula import gpu
gpu.set_gpu(True) # Chebyshev moments on the GPU from now on
(x,y) = h.get_dos(mode="KPM",
            energies=np.linspace(-3.0,3.0,200),
            delta=1e-4,ntries=10,
            kpm_prec="single") # single precision moments
```

When no GPU is available the same jax kernel runs on the CPU with a warning,
so the same script runs on both.

## The local density of states of a large system

We will now see how to compute the local density of states of every site of a large
system, the map that STM measures, with `h.get_multildos()` and `h.get_ldos()` in
`mode="KPM"`. The local density of states of an orbital $i$ is the diagonal element of the
spectral function, $D(\omega,i)=\langle i|\delta(\omega-H)|i\rangle$, and it is expanded in
Chebyshev polynomials as the total one is, with the difference that the moments are now
those of a single orbital

$$
D(\omega,i) = \frac{g_0\mu_0(i) + 2\sum_{m\geq 1} g_m\mu_m(i)\,T_m(\omega/s)}{\pi s\sqrt{1-(\omega/s)^2}},
\qquad \mu_m(i)=\langle i|T_m(H/s)|i\rangle
$$

where $s$ is the scale that brings the spectrum inside $[-1,1]$, $T_m$ are the Chebyshev
polynomials and $g_m$ the damping factors of the Jackson kernel, meaning that the map at
every energy follows from one set of moments per orbital, and that those come from the
Chebyshev recursion started on that orbital. Running that recursion over the whole system
for every orbital costs the square of the number of sites, which is what puts an LDOS map of
$10^5$ sites out of reach. The way we avoid it is by noting that after $m$ steps the vector
started on orbital $i$ has only reached the sites within $m$ hops of it, and that the
$2n_{\rm pol}$ moments of the expansion only need that vector up to $n_{\rm pol}-1$ steps,
since $T_{2m}=2T_mT_m-T_0$ and $T_{2m+1}=2T_{m+1}T_m-T_1$ give two moments from each step.
So the moments computed on the Hamiltonian restricted to the sites within $n_{\rm pol}-1$
hops of an orbital are exactly those of the whole system, and the map costs the number of
sites times the size of that ball, linear in the number of sites

```python
from pyqula import geometry
import numpy as np
g = geometry.honeycomb_lattice().get_supercell(80) # 12800 sites
g.dimensionality = 0 # a finite island with zigzag edges
h = g.get_hamiltonian(is_sparse=True,has_spin=False) # keep it sparse
(x,y,es,ldos) = h.get_multildos(energies=np.linspace(-1.,1.,41), # energies
                    mode="KPM", # Chebyshev expansion of every site
                    delta=0.1, # energy resolution, half width of the peak of a level
                    write=False) # return the maps without writing them
```

`ldos` holds one map per energy, one row for each energy of `es`, all of them from the same
moments, so that 41 energies cost roughly what one does, and `h.get_ldos(e=0.,mode="KPM",
delta=0.1)` returns the row at zero energy alone. For this island the 318 sites at its
zigzag edges, about 2.5% of the sites, carry 56% of the weight at zero energy, which are the
edge states, while at $E=\pm 1$ they carry 1.4%, less than their share, and summing a map
over the sites gives the density of states at that energy, with the normalization of
`h.get_dos(mode="KPM")`. The light cone changes the time of the calculation and not its
result: the map is the one of the expansion over the whole island, to roundoff.

What sets the cost of a site is the resolution. A peak of half width $\delta$ needs about
$1.85\,s/\delta$ polynomials, here 62, and the ball of 61 hops holds about 5700 sites of the
honeycomb lattice, a number that grows as $n_{\rm pol}^2$ in two dimensions, so that halving
$\delta$ makes each site eight times more expensive (twice the steps on four times the
sites), and the map only beats the recursion over the whole system once the sample is larger
than that ball. You can think of it as the statement that a resolution $\delta$ sees the
environment of a site up to a distance of roughly $v/\delta$, with $v$ the velocity of the
states, whatever the size of the sample. The vector carries almost no weight near the edge
of its light cone, so `kpm_radius` below $n_{\rm pol}$ gives a cheaper map that is converged
in practice: on a disordered honeycomb island of 20000 sites at `npol=100`, the map at
`kpm_radius=80`, 70, 60 and 50 differs from the exact one by $10^{-11}$, $2\times10^{-7}$,
$5\times10^{-5}$ and $10^{-3}$ of its largest value, and at 60 it takes about a third of the
time. How far below the light cone one can go depends on how fast the states move in units
of the scale, so the radius is converged as a k-mesh is, by comparing two values.

Optional arguments
- delta: energy resolution, the half width at half maximum of the peak a level gives, as in `h.get_dos(mode="KPM")`
- npol: number of polynomials, in place of `delta`; 200 when neither is given
- kpm_radius: radius in hops of the region the recursion of each site runs on; `None` (default) gives the exact map, a smaller value a cheaper map of a finite cluster around each site
- operator: an operator with a k-independent matrix, giving the local matrix element $\mathrm{Re}\,(A\rho)_{ii}$, the convention of `mode="green"`
- scale: half width of the window that holds the spectrum, estimated from the Hamiltonian when not given
- kpm_prec: precision of the recursion, `"double"` on the CPU and `"single"` on the GPU by default
- nk: k-points per direction, for a periodic Hamiltonian, whose map is averaged over the Brillouin zone

The recursion is the one behind the KPM mean field of "Interactions at the mean-field
level", and it follows the same GPU switch, so a self-consistent state of $10^5$ sites can be
read with the map that STM would measure on it, the spin-resolved one included with
`operator="sz"`. See `examples/0d/kpm_ldos_map/main.py` for a runnable version.
