# Non-Hermitian Hamiltonians

Up to now every Hamiltonian in this guide has been Hermitian, meaning that its
eigenvalues are real energies and its eigenvectors are orthogonal. Let us now
see what happens when the system is open, exchanging particles or energy with
an environment. Such a system is often described by an effective Hamiltonian
that is no longer Hermitian: gain and loss enter as imaginary onsite energies,
and a non-reciprocal hopping, with $t_{ij}\ne t_{ji}^*$, as an asymmetric
hopping matrix. The eigenvalues are then complex, and what their imaginary
parts measure is an amplification or a decay rate rather than an energy, while
the eigenvectors are no longer orthogonal. This is the setting of photonic and
acoustic lattices with gain and loss, of the non-Hermitian skin effect, and of
$\mathcal{PT}$-symmetric models.

pyqula generates such a Hamiltonian with the `non_hermitian=True` flag of
`g.get_hamiltonian()`, after which the usual observables are computed with
their non-Hermitian counterparts. Nothing else about building the model
changes: an `h.add_onsite()` with a complex-valued function is what puts the
gain and loss in. As an example, let us take a chain of twenty sites with a
purely imaginary modulation of the onsite energy, gain on the sites where the
cosine is positive and loss where it is negative

```python
import numpy as np
from pyqula import geometry

n = 20
g = geometry.chain().get_supercell(n)
h = g.get_hamiltonian(has_spin=False,non_hermitian=True,tij=[-1.])
# a purely imaginary Aubry-Andre modulation, with no Hermitian part at
# all: gain where the cosine is positive, loss where it is negative
h.add_onsite(lambda r: 0.6j*np.cos(2.*np.pi*r[0]/n))

(ks,es) = h.get_bands(kpath=[[0.,0.,0.]],write=False)
print("eigenvalues are complex:",es.dtype)
print("largest gain rate Im(E):",np.round(np.max(es.imag),4))
```

The eigenvalues come back complex rather than real: twenty of them, one per
site of the supercell, with real parts inside the band of the chain and
imaginary parts between $-0.48$ and $+0.48$, smaller than the $0.6$ of the
modulation because the hopping mixes the gain and the loss sites. Anything
that plots them has to choose a part: `es.real` for the energy axis, `es.imag`
for the gain and decay axis, or both as a scatter in the complex plane.

## Choosing which part is the energy: eigmode

`eigmode` is the argument that exists only on this path. It says which part of
the complex eigenvalue plays the role of the energy, the quantity a broadening
is centred on or a band structure is written out as, and it takes `"complex"`
(the default, keep the whole eigenvalue), `"real"` or `"imag"`; anything else
raises `ValueError` listing the three.

In `h.get_bands()` it selects what the returned energy row and the written
`BANDS.OUT` carry. With the default `"complex"` the file gains one extra
column, so that its layout is `k`, `Re E`, `Im E`, then one column per
operator; the imaginary part is kept rather than silently dropped, since it is
the physics the calculation was done for.

In `h.get_ldos()` it decides which axis the requested energy `e` lives on.
With `eigmode="imag"` the states are selected by their amplification rate
instead of by their energy, which is how one asks where the most amplified
mode of the model above actually sits

```python
import numpy as np
from pyqula import geometry
n = 20
g = geometry.chain().get_supercell(n)
h = g.get_hamiltonian(has_spin=False,non_hermitian=True,tij=[-1.])
h.add_onsite(lambda r: 0.6j*np.cos(2.*np.pi*r[0]/n))
(ks,es) = h.get_bands(kpath=[[0.,0.,0.]],write=False)

imax = np.argmax(es.imag) # the most amplified state
(x,y,d) = h.get_ldos(e=es[imax].imag,delta=1e-2,eigmode="imag",nrep=1)
print("it lives at x =",x[np.argmax(d)])
```

which returns the site where the gain is largest, $x=-0.5$ here, the maximum
of the modulation above: the most amplified state lives where the imaginary
onsite energy is largest, as one would expect.

`h.get_dos()` broadens the real part whatever `eigmode` says, so that
`"complex"` and `"real"` give the same density of states there and `"imag"`
gives the distribution of decay rates instead.

## The two spectral functions of a non-Hermitian Hamiltonian

We will now see why a non-Hermitian Hamiltonian has two momentum-resolved spectral
functions, and how to ask `h.get_kdos_bands()` for either. In a Hermitian system an
eigenstate is its own dual, so weighing it by an operator is unambiguous. A non-Hermitian
one has right eigenvectors $|R_n\rangle$ and left eigenvectors $\langle L_n|$ that are
different, and the weight of a state can be taken with the right eigenvector alone,
$\langle R_n|O|R_n\rangle/\langle R_n|R_n\rangle$, or with both,
$\langle L_n|O|R_n\rangle/\langle L_n|R_n\rangle$. The first is the default of
`h.get_kdos_bands()`: each state is a Lorentzian of width `delta` at the real part of its
energy, with a weight that is positive for a positive operator, which is the picture of the
band structure one usually draws. The second is what the Green's function gives

$$
A(\vec k,\omega) = -\frac{1}{\pi}\,\mathrm{Im}\,\mathrm{Tr}\left[O\,(\omega+i\delta-H_{\vec k})^{-1}\right]
= -\frac{1}{\pi}\,\mathrm{Im}\sum_n \frac{\langle L_n|O|R_n\rangle/\langle L_n|R_n\rangle}{\omega+i\delta-E_n}
$$

meaning that a state whose energy has an imaginary part $\mathrm{Im}\,E_n<0$ is broadened
by it, its finite lifetime, which is how an effective non-Hermitian Hamiltonian describes
quasiparticles that decay, and the spectral function an ARPES measurement of such
quasiparticles sees. `biorthogonal=True` asks for it

```python
import numpy as np
from pyqula import geometry
g = geometry.chain().get_supercell(4,store_primal=True) # supercell of a chain
h = g.get_hamiltonian(has_spin=False,non_hermitian=True) # non-Hermitian Hamiltonian
h.add_onsite(lambda r: -0.3j*(1.+np.cos(np.pi*r[0]/2.))) # modulated loss
kpath = g.get_unfolded_kpath(nk=100) # primitive k-path, in supercell coordinates
(k,e,d) = h.get_kdos_bands(operator="unfold",kpath=kpath,
        biorthogonal=True) # unfolded Green's-function spectral function
```

Compared with the default, each unfolded band comes out broader and lower, by the loss of
the states it is made of. `mode="green"` computes the same numbers from the Green's function
directly, which is why it refuses a non-Hermitian Hamiltonian unless `biorthogonal=True` is
passed, so that changing the mode never changes which spectral function comes out, and
`mode="KPM"` refuses one altogether, for the reason `h.get_dos()` does. The two spectral
functions agree when the spectrum is real, and they differ in two ways otherwise: the Green's
function one carries the lifetime of each state, and its weight can be negative for a state
with gain, $\mathrm{Im}\,E_n>0$, whose pole sits on the wrong side of the real axis. Close to
an exceptional point, where two eigenvectors coalesce, the biorthogonal weights of the two
states grow large and cancel each other, and only their sum is meaningful.
`h.get_bands(operator=...,biorthogonal=True)` returns those weights state by state, complex
in general, adding up to the trace of the operator at every k-point, and
`h.get_dos(biorthogonal=True)` sums the same spectral function over the Brillouin zone.

## What is and is not available

The band structure, the density of states, the LDOS and the Berry curvature
have genuine non-Hermitian versions, and so does the density matrix. The
occupied states are those with the lowest real part of the energy, and each of
them enters with its right eigenvector $|R\rangle$ and its left eigenvector
$\langle L|$, normalized so that $\langle L|R\rangle=1$, so that
$\rho=\sum_{\rm occ}|R\rangle\langle L|$ and an expectation value is the sum of
the biorthogonal ones $\langle L|A|R\rangle$, complex in general.
`h.get_density_matrix()` is built from this $\rho$, `h.get_vev()` and
`h.get_magnetization()` return the real part of what it gives, and the self-consistent
mean field is solved with it, so that the mean-field Hamiltonian is itself
non-Hermitian. This works for a normal-state Hamiltonian with exact
diagonalization, the default, and the KPM, `use_jax=True` and
`integration="qtci"` refuse a non-Hermitian Hamiltonian with a
`NotImplementedError`. A simple check of all this is the open Hatano-Nelson
chain, whose non-reciprocal hopping an imaginary gauge transformation turns
into the Hermitian hopping $\sqrt{t_Rt_L}$ without changing any density, so
that with a Hubbard $U$ its self-consistent spectrum is that of the Hermitian
chain (see
`jupyter-notebooks/functionalities/interacting_mean_field_hamiltonians/08_hermitian_nonhermitian.ipynb`).
Everything else on a `non_hermitian=True` Hamiltonian falls back to the
Hermitian formula, which may or may not be meaningful for a complex spectrum,
so check before relying on it.

One restriction is enforced rather than left to the caller: `h.get_dos()`
accepts only `mode="ED"`, and refuses `use_kpm=True` and every other mode with
a `NotImplementedError`, because the Chebyshev and adaptive expansions both
assume a real spectrum. `h.get_ldos()` has the same limitation, only its
default diagonalization mode (`mode="arpack"`) is implemented here, but does
not enforce it, so pass no other mode.

Operators work as usual, including `operator="unfold"`, so that a supercell
calculation can be unfolded back onto the primitive Brillouin zone with a
complex spectrum; see `examples/1d/unfolding_non_hermitian/main.py`.
`num_bands` also works, selecting the few eigenvalues nearest `central_energy`
rather than diagonalizing fully.

See `examples/1d/NH_ldos/main.py` (the model above, resolved mode by mode),
`examples/0d/non_hermitian_aah/main.py` and
`examples/0d/non_hermitian_aah_dos/main.py` (a non-Hermitian Aubry-Andre chain
swept over the modulation phase) for runnable versions, and
`jupyter-notebooks/functionalities/single_particle_hamiltonians/09_non_hermitian_hamiltonians.ipynb`
for an executed notebook.
