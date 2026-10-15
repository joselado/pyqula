# Quasiparticle interference

In this page we compute what a single defect does to the local density of
states of a two-dimensional system. A defect scatters an electron from a
state with momentum $\mathbf k$ to a state with momentum $\mathbf k'$ on the
same constant-energy contour. The incoming and the scattered waves interfere,
so the local density of states around the defect oscillates with wavevector
$\mathbf q = \mathbf k' - \mathbf k$. The Fourier transform of these
oscillations at each energy is called the quasiparticle interference (QPI)
pattern. In an experiment, it is obtained by STM as the Fourier transform of
a conductance map $dI/dV(\mathbf r,V)$. It shows the constant-energy contours
of the band structure, and which pairs of states the defect connects.

## QPI from the spectral function

The simplest estimate of the QPI pattern is the autocorrelation of the
momentum-resolved spectral function,

$$
P(\mathbf q,\omega) = \sum_{\mathbf k} A(\mathbf k,\omega)\,A(\mathbf k+\mathbf q,\omega) ,
$$

where $A(\mathbf k,\omega)$ is the spectral weight at momentum $\mathbf k$ and
energy $\omega$, broadened with a Lorentzian of width $\delta$ (see
[Spectral functions](spectral-functions.md)). It counts all the pairs of
states at energy $\omega$ separated by $\mathbf q$, so it is large for a
$\mathbf q$ that connects two flat, parallel parts of the contour. This is
`mode="pm"` of `h.get_qpi()`, for 2D Hamiltonians:

```python
import numpy as np
from pyqula import geometry
g = geometry.triangular_lattice()
h = g.get_hamiltonian(has_spin=False)
(qs,es,qpi) = h.get_qpi(mode="pm",nk=50,delta=1e-1,energies=np.linspace(-6.,6.,100)) # QPI at each energy
```

`qs` contains the wavevectors $\mathbf q$, one per row, `es` the energies,
and `qpi` the intensity, with one row per energy and one column per
wavevector. So one row of `qpi`, plotted over `qs`, is the QPI map of an STM
measurement at that energy. The band of the triangular lattice goes from
$-3t$ at K and K' to $6t$ at $\Gamma$. So near the top of the band the
contour is a small circle of radius $k_F$ around $\Gamma$, and the QPI map
is a disk of radius $2k_F$, since $2k_F$ is the largest momentum that
connects two points of the circle. Near the bottom of the band the contour
is two small pockets, around K and K'. The same maps are written to the folder `MULTIQPI/`, one file
per energy, with the density of states in `DOS.OUT`, unless `write=False` is
given.

![QPI maps of the triangular lattice, at an energy near the top of the band and at an energy near the bottom.](img/qpi_pm.png#only-light)
![QPI maps of the triangular lattice, at an energy near the top of the band and at an energy near the bottom.](img/qpi_pm-dark.png#only-dark)

The wavevectors are Cartesian, in units of $2\pi/a$, and the color scale
stops at the 99th percentile, because the peak at $\mathbf q=0$ is much
larger than the rest. With this sign of the hopping, the band of the
triangular lattice goes from $-3t$ at K to $6t$ at $\Gamma$, so the small
circle around $\Gamma$ is at the top of the band. At $\omega=5.03t$ (left)
the contour is a circle of radius $k_F\approx 0.13$ and the QPI map is a
disk of radius $0.27$, which is $2k_F$, repeated at every reciprocal lattice
vector. At $\omega=-2.48t$ (right), near the bottom of the band, the contour
is two small pockets around K and K'. The map has a disk at $\mathbf q=0$,
from the scattering inside one pocket, and bright features at the K points of
$\mathbf q$, from the scattering between the two pockets.

??? example "Code of this figure (documentation/figures/qpi_pm.py)"

    ```python
    --8<-- "qpi_pm.py"
    ```

The keywords are

- `energies`: the energies $\omega$.
- `nk`: number of k-points per direction.
- `delta`: the broadening $\delta$.
- `mode`: `"pm"`, the autocorrelation above, or `"response"` (default), a
  nesting function of the clean bands, defined below.
- `operator`: with `"unfold"`, the spectral function of a supercell is
  unfolded to the primitive Brillouin zone (`mode="pm"` only).
- `nunfold`: the linear size of the supercell for the unfolding, as in
  `g.get_supercell()`: $n$ for an $n\times n$ supercell, `np.sqrt(3)` for a
  $\sqrt3\times\sqrt3$ one.
- `write`: `False` to return the maps without writing files.

`mode="response"` uses only the energies of the bands, not the
wavefunctions,

$$
R(\mathbf q,\omega) = \sum_{\mathbf k,\alpha,\beta}
\Theta(\omega-\epsilon_{\mathbf k+\mathbf q,\alpha})\,
\Theta(\epsilon_{\mathbf k,\beta}-\omega)\,
\frac{\delta}{(\epsilon_{\mathbf k+\mathbf q,\alpha}-\epsilon_{\mathbf k,\beta})^2+\delta^2} ,
$$

which counts pairs of states, one below and one above $\omega$, with almost
the same energy. It is a measure of the nesting of the contour at energy
$\omega$. It is cheaper, but it ignores the form factors of the
wavefunctions, and it does not accept an operator or unfolding.
`mode="pm"` uses the real spectral weight of the system, defect included,
and it is the one to use for a real scatterer.

The realistic case is a single point defect in a supercell, with the QPI
unfolded back to the Brillouin zone of the primitive cell. The supercell
must be built with `store_primal=True`, so that the primitive cell is known
for the unfolding:

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

This calculation is expensive. `mode="pm"` diagonalizes the Hamiltonian on an
`nk` x `nk` mesh and then computes the autocorrelation, so `nk=140` takes
minutes. Use `nk=60` (about 40 seconds) to prepare the calculation, and a
large `nk` only for the final result, since `nk` sets the resolution in
$\mathbf q$.

See `examples/2d/multiqpi/main.py` (clean system, `mode="pm"`) and
`examples/2d/multiqpi_unfold/main.py` (defect in a supercell, unfolded) for
runnable scripts.

## QPI from a real-space impurity

`h.get_qpi_impurity()` does the calculation in the same way as the
experiment. It builds a supercell of `h`, puts one or more impurities in it,
computes the local density of states $D(\mathbf r,\omega)$ around them (see
[Local density of states](ldos.md)), and takes its Fourier transform,

$$
\textrm{QPI}(\mathbf q,\omega) =
\left| \sum_{\mathbf r} e^{-i\mathbf q\cdot\mathbf r}\, D(\mathbf r,\omega) \right| .
$$

So here the interference of the scattered waves is included exactly, for the
impurity potential you choose:

```python
from pyqula import geometry
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False)
r,ldos_r,q,qpi_q = h.get_qpi_impurity(nsuper=10,
        impurities=[{"position": [0.,0.,0.], "onsite": 3.0}],
        energies=0.3,num_waves=60,nk=2,delta=0.2)
```

`r` and `ldos_r` are the positions of the sites and the local density of
states on them, the real-space map. `q` and `qpi_q` are the wavevectors and
the QPI intensity, its Fourier transform.

![Change of the local density of states around an impurity in the honeycomb lattice, and its Fourier transform.](img/qpi_impurity.png#only-light)
![Change of the local density of states around an impurity in the honeycomb lattice, and its Fourier transform.](img/qpi_impurity-dark.png#only-dark)

For the figure we take `nsuper=12` instead of `nsuper=10`, because the K
points are on the $\mathbf q$ mesh only when the size is a multiple of
three. Lengths are in units of the first-neighbor distance $a$. On the left,
the change of the local density of states at $\omega=0.3t$ with respect to
its mean. The color scale stops at the 98th percentile, so that the impurity
site, where the density drops, does not hide the rest. You see the Friedel
oscillations around the impurity, with a period of three sites. On the
right, the QPI intensity on the 144 points of the mesh. The point at
$\mathbf q=0$ is left out (white), since it is the mean density and it is
300 times larger than the rest. The largest values are around the K points,
at $|\mathbf q|\approx 2.4/a$, at a distance of about $0.3$ to $0.6/a$ from
them, close to $2k_F=0.4/a$ of the Dirac cone at this energy. This is the
scattering between the two valleys, and it gives the period of three sites
in real space.

??? example "Code of this figure (documentation/figures/qpi_impurity.py)"

    ```python
    --8<-- "qpi_impurity.py"
    ```

The keywords are

- `nsuper`: size of the supercell, a number or `(n1,n2)`.
- `impurities`: a list of impurities. Each one is an onsite potential,
  `{"position": [x,y,z], "onsite": v}` (or `{"index": i, "onsite": v}` for
  a site of the supercell), or a vacancy,
  `{"position": [x,y,z], "vacancy": True}`.
- `energies`: one energy or an array of energies.
- `num_waves`: a first guess for the number of eigenstates computed near
  each energy. It is increased automatically when needed.
- `nk`, `delta`: as in `h.get_ldos()`.

A vacancy is modeled as a very large onsite potential on the site, which in
practice removes the site from the low-energy states.

The wavevectors `q` are the $n_1\times n_2$ points that are commensurate with
the supercell, and they cover the whole Brillouin zone of the primitive cell.
So `nsuper` sets the resolution in $\mathbf q$, not the range. Only these
points are free of finite-size artifacts: at any other $\mathbf q$, even a
clean supercell would give a signal. No unfolding is needed here, because the
Fourier transform is done directly on the real-space density.

See `examples/2d/qpi_realspace_impurity/main.py` for a runnable script that
plots the real-space map and the QPI, and
`jupyter-notebooks/functionalities/spectral_functions/07_qpi_maps.ipynb` for
an executed notebook.
