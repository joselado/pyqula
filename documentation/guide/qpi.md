# Quasiparticle interference

## Quasiparticle interference

Let us now see what a single defect does to the local density of states of a
two-dimensional system, with `h.get_qpi()`. A defect scatters an electron from a state at
momentum $\vec k$ to a state at $\vec k'$ on the same constant-energy contour, and the two
interfere, so that the local density of states around the defect is modulated with
wavevector $\vec q = \vec k - \vec k'$. The Fourier transform of that modulation at each
energy is what is called the quasiparticle interference pattern, and it is what an STM
measurement obtains by Fourier transforming a conductance map: a picture of the
constant-energy contours of the band structure, and of which pairs of states the defect
connects. `h.get_qpi()` computes it for 2D Hamiltonians

```python
import numpy as np
from pyqula import geometry
g = geometry.triangular_lattice()
h = g.get_hamiltonian(has_spin=False)
(qs,es,qpi) = h.get_qpi(mode="pm",nk=50,delta=1e-1,energies=np.linspace(-6.,6.,100)) # QPI at each energy
```

where `qs` holds the wavevectors $\vec q$, one per row, `es` the energies, and `qpi` the
intensity of the pattern, one row per energy and one column per wavevector, so that a row of
`qpi` plotted over `qs` is the map an STM measurement gives at that energy. The same maps are
also written, one file per energy, to a folder (`MULTIQPI/` by default), with the density of
states next to it in `DOS.OUT`, unless `write=False` is passed

Optional arguments
- energies: array of energies to compute
- nk: number of k-points per direction
- delta: broadening
- mode: `"pm"` (autoconvolution of the k-resolved spectral weight) or `"response"` (default, joint density of states of the clean band structure)
- nunfold: unfold the QPI of a supercell back onto the primitive Brillouin zone, given as the linear size of the supercell as in `g.get_supercell()`, $n$ for an $n\times n$ supercell and `np.sqrt(3)` for the $\sqrt3\times\sqrt3$ one
- write: `False` to return the maps without writing the folder and `DOS.OUT`

The two modes differ in what they scatter. `"pm"` ("poor man's") autoconvolves the actual
k-resolved spectral weight of the system, defect included, in q-space, which is the
physically meaningful QPI signal for a real scatterer; `"response"` instead computes a
cheaper Lindhard-like convolution of the joint density of states from the clean band
structure only, ignoring the form factors of the wavefunctions.

A single point defect embedded in a supercell, with the resulting QPI unfolded back onto the
primitive cell, is the realistic case. The supercell must be built with `store_primal=True`
so that pyqula remembers the primitive cell needed to unfold; `operator="unfold"` then
resolves to the corresponding unfolding operator

```python
import numpy as np
from pyqula import geometry
g0 = geometry.honeycomb_lattice()
ns = 2
g = g0.get_supercell(ns,store_primal=True)
h = g.get_hamiltonian(has_spin=False)
h.add_onsite(lambda r: 100.0 if np.linalg.norm(r-g.r[0])<1e-1 else 0.0) # a strong point defect

h.get_qpi(mode="pm",delta=1e-2,operator="unfold",nsuper=2,nk=140,nunfold=ns)
```

This is the most expensive snippet in the guide: `mode="pm"` diagonalizes on an `nk`x`nk`
mesh and then autoconvolves the result, so the cost grows quadratically with `nk` and the
`nk=140` above takes minutes. Drop to `nk=60` (about 40 seconds) while setting a calculation
up, and raise `nk` only for the final figure, since the q-space resolution of the QPI pattern
is what it buys.

See `examples/2d/multiqpi/main.py` (clean system, `mode="pm"`) and `examples/2d/multiqpi_unfold/main.py` (defect in a supercell, unfolded) for runnable versions.

### Real-space-impurity QPI

The modes of `h.get_qpi()` all work in reciprocal space: they convolve or scatter k-resolved
spectral weight and never touch a real-space impurity. `h.get_qpi_impurity()` takes the
direct route instead, the one an experiment takes: it builds a supercell of `h`, puts one or
more actual impurities in it, computes the real-space LDOS map around them, and Fourier
transforms that map to obtain the QPI signal as a function of $\vec q$. Unlike `get_qpi()`,
it returns arrays rather than only writing to disk

```python
from pyqula import geometry
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False)
r,ldos_r,q,qpi_q = h.get_qpi_impurity(nsuper=10,
        impurities=[{"position": [0.,0.,0.], "onsite": 3.0}],
        energies=0.3,num_waves=60,nk=2,delta=0.2)
```

The four arrays are the positions of the sites and the LDOS on them, and the q-points and the
QPI weight on them, so the first pair is the real-space map and the second its Fourier
transform.

Optional arguments
- nsuper: supercell size (scalar or `(n1,n2)`)
- impurities: list of dicts, each an onsite potential (`{"position": [x,y,z], "onsite": v}`, or `{"index": i, "onsite": v}` for a specific supercell site index) or a vacancy (`{"position": [x,y,z], "vacancy": True}`)
- energies: a single energy or an array
- num_waves: a starting guess for how many eigenstates nearest the requested energies to compute
- nk, delta: as in `get_ldos`

A vacancy is modeled as a strong onsite potential rather than as a removed site, since
deleting sites from a large sparse supercell Hamiltonian would require densifying it. The
number of eigenstates `num_waves` is only a starting guess: it is grown automatically until
the window covers `margin*delta` past every requested energy and never cuts a degenerate
manifold in half, because summing over half a degenerate manifold is not basis independent
and would leak spurious QPI weight even for a clean supercell.

No unfolding step is needed here: this route never diagonalizes supercell bands and projects
them onto primitive Bloch states (which is what `get_qpi`'s `nunfold` and `store_primal` are
for). It only Fourier transforms a real-space scalar density, evaluated directly at q spanning
the full primitive Brillouin zone. `q` is fixed at exactly the `nsuper1`x`nsuper2` points
commensurate with the supercell, meaning that `nsuper` sets the achievable q resolution, not
the range of the Brillouin zone; evaluating the direct-sum Fourier transform at any other q
would show finite-size leakage even for a perfectly clean system, since only the commensurate
points are free of it.

See `examples/2d/qpi_realspace_impurity/main.py` for a runnable version that plots both the real-space LDOS and QPI(q), and `jupyter-notebooks/functionalities/spectral_functions/07_qpi_maps.ipynb` for an executed notebook.
