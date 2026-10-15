# Response functions

Let us now see how a quantum material responds to a perturbation: a field
that couples to an operator $B$ is switched on, the change of an operator $A$
is measured, and their ratio as a function of frequency and momentum transfer
is the response function $\chi_{AB}(q,\omega)$, whose imaginary part is what
a spectroscopy measures. We will build it up in three steps: the bare
response of a single-particle Hamiltonian, a sum over transitions between its
eigenstates; the same response dressed with the interaction of a mean-field
state at the random phase approximation (RPA) level, whose poles are the
collective modes, magnons and plasmons; and the Bethe-Salpeter equation,
which solves the two-particle problem directly and gives the excitons of a
semiconductor. The mean-field chapter feeds the last two steps.

## Optical conductivity

We will start with the response to an electric field, the frequency-dependent
conductivity tensor of a periodic Hamiltonian, with
`h.get_optical_conductivity()`. It is computed with the Kubo-Greenwood
formula, summing velocity matrix elements over a k-mesh in the Lehmann
representation,

$$
\sigma_{ab}(\omega) = \frac{i e^2 \hbar}{N_k V_{\rm cell}} \sum_{\mathbf k}
\sum_{n \neq m} \frac{f_n - f_m}{E_m - E_n}
\frac{v^a_{nm} v^b_{mn}}{\hbar\omega + i\eta - (E_m-E_n)}
$$

where $v^a_{nm}$ is the matrix element of the velocity along $a$ between the
bands $n$ and $m$, $f_n$ the occupation of band $n$ and $\eta$ a broadening,
following the Wannier90 postw90 convention (Yates, Wang, Vanderbilt and
Souza, Phys. Rev. B 75, 195121): light of frequency $\omega$ is absorbed
whenever it can move an electron from an occupied to an empty band separated
by $\hbar\omega$, with a weight set by the velocity matrix element. The full
complex tensor is returned, so $\mathrm{Re}\,\sigma_{xx}$ is the optical
absorption, $\sigma_{xy}$ the magneto-optical (Kerr and Faraday) response,
and the $\omega \to 0$ limit of $\sigma_{xy}$ the anomalous Hall
conductivity, quantized to $-C\,e^2/h$ for a Chern insulator with Chern
number $C$. The intraband (Drude) channel comes from the degenerate limit
$(f_n-f_m)/(E_m-E_n) \to -\partial f/\partial E$ of the same sum, which also
protects the formula against $0/0$ on spin-degenerate multiplets;
`intraband` and `interband` switch the two channels independently.

Results are in units of $e^2/\hbar$, so one conductance quantum $e^2/h$ is
$1/(2\pi)$ of the returned value, times $a^{2-d}$ for a $d$-dimensional
lattice. `T` sets the temperature and `delta` the Lorentzian broadening
$\eta$. The absolute normalization is fixed by the f-sum rule,
$\int \mathrm{Re}\,\sigma_{aa}(\omega)\,d\omega = \pi W_{aa}$, with $W$ the
diamagnetic weight available as `h.get_sum_rule_weight()`; the Drude weight
tensor is `h.get_drude_weight()`. Let us take a Chern insulator, the
honeycomb lattice with a Haldane term, and compute both its absorption and
its Hall response

```python
import numpy as np
from pyqula import geometry
g = geometry.honeycomb_lattice() # create a honeycomb lattice
h = g.get_hamiltonian(has_spin=False) # get the Hamiltonian
h.add_haldane(0.2) # Chern insulator
h.shift_fermi(0.3) # put the Fermi energy in the gap

(ws,s) = h.get_optical_conductivity(energies=np.linspace(0.,4.,100),
                                    nk=40,T=0.02,delta=0.05)
absorption = s[:,0,0].real # Re sigma_xx, the optical absorption

# the DC Hall response is quantized to the Chern number
(w0,s0) = h.get_optical_conductivity(energies=[0.],nk=40,T=0.01,delta=1e-3)
print("sigma_xy(0) in e^2/h:",2.*np.pi*s0[0,0,1].real) # -1
```

`ws` holds the frequencies and `s` the $3\times3$ tensor at each of them.
What you see in the absorption is zero below the gap and an onset at it; the
second call evaluates the tensor at zero frequency with a small broadening,
and the off-diagonal element comes out at one conductance quantum with the
sign of the Chern number. The velocity operator behind the formula is the
current operator built from the full bond vectors, including the bonds inside
the unit cell, so that the tensor has the symmetry of the lattice: on the
honeycomb lattice $\sigma_{xx} = \sigma_{yy}$ to machine precision, and
graphene reproduces its universal absorption $\pi e^2/4h$. Superconducting
(Nambu) and 3d Hamiltonians raise `NotImplementedError`. See
`examples/2d/optical_conductivity/main.py` and
`examples/1d/optical_conductivity_chain/main.py` for runnable versions, and
`jupyter-notebooks/functionalities/response_functions/01_optical_conductivity.ipynb` and
`02_drude_weight_f_sum_rule.ipynb` in the same folder for executed notebooks.
