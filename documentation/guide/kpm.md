# Kernel polynomial method

Exact diagonalization gives all the eigenstates, and its cost grows as the
cube of the number of sites. For a system with $10^5$ or $10^6$ sites this
is not possible, and also not needed: the density of states is a trace, and
a trace can be computed without the eigenstates. The kernel polynomial
method (KPM) does this. It expands the density of states in Chebyshev
polynomials of the Hamiltonian, and only needs products of a vector with the
sparse Hamiltonian, so its cost is linear in the number of sites.

## The density of states of a large system

The Chebyshev polynomials $T_m(x)$ are defined on $[-1,1]$, so we first
rescale the Hamiltonian, $\tilde H = H/s$, with a scale $s$ larger than the
largest energy. They obey the recursion

$$T_0(x)=1,\qquad T_1(x)=x,\qquad T_{m+1}(x)=2xT_m(x)-T_{m-1}(x).$$

The density of states at energy $E$, with $x=E/s$, is

$$
\rho(E) = \frac{1}{\pi s\sqrt{1-x^2}}
\left[ g_0\mu_0 + 2\sum_{m\geq 1} g_m\,\mu_m\,T_m(x) \right],
\qquad
\mu_m = {\rm Tr}\, T_m(\tilde H) .
$$

The $\mu_m$ are the moments. They are computed with the recursion above,
applied to a vector: $|v_{m+1}\rangle = 2\tilde H|v_m\rangle - |v_{m-1}\rangle$,
so each moment costs one sparse matrix-vector product. The trace is
estimated with random vectors $|r\rangle$, as
${\rm Tr}\,A \approx \frac{1}{R}\sum_{r=1}^{R}\langle r|A|r\rangle$, and the
error goes down as $1/\sqrt R$. The series is cut after $N$ moments, and
this creates oscillations (Gibbs oscillations). The factors $g_m$ of the
Jackson kernel remove them,

$$
g_m = \frac{(N-m+1)\cos\frac{\pi m}{N+1} + \sin\frac{\pi m}{N+1}\cot\frac{\pi}{N+1}}{N+1},
$$

and the price is that every eigenvalue becomes a peak of finite width,
nearly a Gaussian. In pyqula $N=2n_{\rm pol}$, with $n_{\rm pol}$ the number
of polynomials, because each step of the recursion gives two moments.

The KPM density of states is `h.get_dos()` with `mode="KPM"`:

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

`x` is the energy grid and `y` the density of states, the same quantity that
exact diagonalization gives for this chain of 3000 sites. For a chain you
see the one-dimensional van Hove singularities at $E=\pm2$. A sparse
Hamiltonian (`is_sparse=True`) is the natural choice, since the method never
needs the dense matrix.

Two numbers control the result:

- `delta`: the energy resolution, the half width at half maximum of the peak
  of a single level. It is the same width that `delta` gives as a Lorentzian
  in `mode="ED"`, so the two curves can be compared. The method uses
  $n_{\rm pol}\approx 1.85\,s/\delta$ polynomials, so a sharper density of
  states costs more. The KPM peak is nearly a Gaussian, about 1.5 times
  taller than the Lorentzian and without its long tails.
- `ntries`: the number of random vectors $R$. The noise in `y` goes as
  $1/\sqrt R$.

`delta` must be larger than the distance between levels, here
$2\pi t/N\approx 0.002$ in the middle of the band, otherwise `y` shows the
single levels of the finite chain. The spectrum must be inside $[-s,s]$,
since $T_m(x)$ grows exponentially outside $[-1,1]$. The default is
`scale=10` in units of the hopping. A Hamiltonian with a larger spectrum, for
example with a large onsite energy or exchange field, needs a larger
`scale`. If `scale` is too small, the code raises a `ValueError`.

### Green's functions between two sites

The same expansion also gives the Green's function between two sites $i$
and $j$ of this chain, without inverting a matrix. The function
`kpm.dm_ij_energy()` gives

$$
\pi\,\langle i|\delta(E-H)|j\rangle = \pi\sum_n \psi_n(i)\psi_n^*(j)\,\delta(E-E_n)
= \frac{i}{2}\left[G^R_{ij}(E)-G^A_{ij}(E)\right],
$$

the spectral part of the Green's function:

```python
from pyqula import kpm
(x,y) = kpm.dm_ij_energy(h.intra,npol=200,i=0,j=9,ne=1000)
```

For $i=j$ it is $\pi$ times the local density of states of site $i$. For
$i\neq j$ it integrates to zero over all energies, because the eigenstates
form a complete basis. Its integral up to the Fermi energy is $\pi$ times the
density matrix element $\langle c^\dagger_j c_i\rangle$, so its sign along the
energy axis tells which part of the spectrum correlates the two sites. Here
`npol` is the number of polynomials $n_{\rm pol}$, and each level becomes a
peak of half width $1.85\,s/n_{\rm pol}$, the inverse of the relation for
`delta` above.

The full Green's function, with its real part, comes from
`kpm.correlator0d()`. It returns the energies, ${\rm Re}\,G^R_{ij}$ and
$-{\rm Im}\,G^R_{ij}$. With `kernel="lorentz"` each pole becomes a
Lorentzian of half width $1.5\,s/n_{\rm pol}$, which is the
$G^R_{ij}(E+i\delta)$ that a matrix inversion gives.

See `examples/0d/kpm_dos/main.py` and `examples/0d/kpm_correlator/main.py`
for runnable scripts, the second one with a comparison against the exact
Green's function, and
`jupyter-notebooks/functionalities/chebyshev_kpm_algorithms/` for the local
and non-local spectral functions of a system too large to diagonalize.

### On a GPU

The Chebyshev moments of the same density of states can also be computed on
a GPU, with the switch described in
[GPU](gpu.md). The keyword `kpm_prec` chooses single or double precision for
the moments:

```python
from pyqula import gpu
gpu.set_gpu(True) # Chebyshev moments on the GPU from now on
(x,y) = h.get_dos(mode="KPM",
            energies=np.linspace(-3.0,3.0,200),
            delta=1e-4,ntries=10,
            kpm_prec="single") # single precision moments
```

If there is no GPU, the same code runs on the CPU with a warning.

## The local density of states of a large system

We now compute the local density of states of every site of a large system,
which is the map that STM measures, with `h.get_multildos()` and
`h.get_ldos()` in `mode="KPM"`. The local density of states of orbital $i$ is
$D(\omega,i)=\langle i|\delta(\omega-H)|i\rangle$. It has the same expansion
as the total density of states, but with the moments of one orbital,

$$
D(\omega,i) = \frac{g_0\mu_0(i) + 2\sum_{m\geq 1} g_m\mu_m(i)\,T_m(\omega/s)}{\pi s\sqrt{1-(\omega/s)^2}},
\qquad \mu_m(i)=\langle i|T_m(H/s)|i\rangle ,
$$

so no random vectors are needed: the recursion starts from orbital $i$.

Doing this recursion over the whole system for every orbital costs the
square of the number of sites. pyqula avoids this. After $m$ steps, the
vector that started on orbital $i$ has only reached the sites within $m$
hops of $i$. The $2n_{\rm pol}$ moments only need $n_{\rm pol}-1$ steps,
because $T_{2m}=2T_mT_m-T_0$ and $T_{2m+1}=2T_{m+1}T_m-T_1$ give two moments
per step. So the moments computed on the sites within $n_{\rm pol}-1$ hops of
$i$ are exactly those of the whole system, and the full map costs a time
linear in the number of sites.

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

`ldos` contains one map per energy, one row for each energy of `es`. All the
rows come from the same moments, so 41 energies cost about the same as one.
`h.get_ldos(e=0.,mode="KPM",delta=0.1)` gives only the map at zero energy.
In this island the 318 sites at the zigzag edges, about 2.5% of all the
sites, carry 56% of the weight at zero energy: these are the zigzag edge
states. At $E=\pm1$ the same sites carry only 1.4%. The sum of a map over the
sites is the density of states at that energy, with the normalization of
`h.get_dos(mode="KPM")`. The result is the same as the expansion over the
whole island, up to roundoff.

The cost of one site is set by the resolution. A peak of half width $\delta$
needs about $1.85\,s/\delta$ polynomials, here 62, and the region of 61 hops
around a site contains about 5700 sites of the honeycomb lattice. In two
dimensions this region grows as $n_{\rm pol}^2$, so halving $\delta$ makes
each site eight times more expensive. You can think of it in this way: with
a resolution $\delta$, a site only sees its environment up to a distance of
about $v/\delta$, where $v$ is the velocity of the electrons, whatever the
size of the sample.

The vector has almost no weight near the border of this region, so
`kpm_radius` smaller than $n_{\rm pol}$ gives a cheaper map that is still
converged in practice. On a disordered honeycomb island of 20000 sites with
`npol=100`, the map with `kpm_radius=80`, 70, 60 and 50 differs from the exact
one by $10^{-11}$, $2\times10^{-7}$, $5\times10^{-5}$ and $10^{-3}$ of its
largest value, and with 60 it takes about one third of the time. How small
the radius can be depends on the velocity of the states, so you should
converge it as you converge a k-mesh, by comparing two values.

The keywords are:

- `delta`: energy resolution, the half width at half maximum of the peak of a level, as in `h.get_dos(mode="KPM")`.
- `npol`: number of polynomials, instead of `delta`; 200 if neither is given.
- `kpm_radius`: radius in hops of the region around each site; `None` (default) gives the exact map.
- `operator`: an operator $A$ independent of k, giving the local matrix element $\mathrm{Re}\,(A\rho)_{ii}$, for example `operator="sz"` for the spin-resolved map.
- `scale`: half width $s$ of the energy window; estimated from the Hamiltonian if not given.
- `kpm_prec`: precision, `"double"` on the CPU and `"single"` on the GPU by default.
- `nk`: k-points per direction for a periodic Hamiltonian; the map is averaged over the Brillouin zone.

The same recursion is used by the KPM mean field of
[Mean-field Hubbard models](mean-field.md), so a self-consistent state of
$10^5$ sites can be read with the map that STM would measure on it. See
`examples/0d/kpm_ldos_map/main.py` for a runnable script.
