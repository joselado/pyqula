# Quantum transport

In this page we compute the conductance of junctions. We start with two
normal leads, then a normal lead and a superconductor, and we end with the
current through a junction between two superconductors at finite voltage.

A transport calculation has a scattering region between two semi-infinite
leads. Each lead is a periodic Hamiltonian, of the kind built in
[Building a Hamiltonian](hamiltonian.md). The leads enter through their
self-energies $\Sigma_L(E)$ and $\Sigma_R(E)$, computed from the surface
Green's function of each lead, and the Green's function of the scattering
region is

$$G(E) = \left[E + i\delta - H_C - \Sigma_L(E) - \Sigma_R(E)\right]^{-1}.$$

For normal leads, the zero-temperature conductance is the Landauer formula,

$$G = \frac{e^2}{h}\, T(E), \qquad
T(E) = {\rm Tr}\left[\Gamma_L\, G\, \Gamma_R\, G^\dagger\right], \qquad
\Gamma_{L,R} = i\left(\Sigma_{L,R} - \Sigma_{L,R}^\dagger\right),$$

where $T(E)$ is the transmission, the sum of the transmission probabilities
of all the channels. pyqula gives conductances in units of $e^2/h$, and
counts the two spins as two channels, so a perfect spinful chain has $G=2$.
This is the differential conductance $dI/dV$ at bias $V=E/e$ that a
two-terminal measurement gives.

When one lead is superconducting, the Hamiltonian is written in the Nambu
basis (see [Superconductivity](superconductivity.md)), and the conductance
is computed from the scattering matrix with the Blonder-Tinkham-Klapwijk
formula,

$$G(E) = \frac{e^2}{h}\left[N - R_{ee}(E) + R_{he}(E)\right],$$

where $N$ is the number of channels of the normal lead,
$R_{ee}={\rm Tr}\, r_{ee}^\dagger r_{ee}$ is the probability that an electron
is reflected as an electron, and $R_{he}={\rm Tr}\, r_{he}^\dagger r_{he}$
the probability that it is reflected as a hole. The second process is
Andreev reflection: an electron from the normal side is reflected as a hole,
and a Cooper pair enters the superconductor. It transfers a charge $2e$, so
it adds to the conductance.

## Magnetoresistance in metal-metal transport

We now compute the magnetoresistance of a junction between two magnetic
metals, with `HT.didv()`. We make two copies of the same chain, add an
exchange field to each of them, and compare the parallel and antiparallel
configurations:

```python
from pyqula import geometry
from pyqula import heterostructures
import numpy as np
g = geometry.chain() # create the geometry
h = g.get_hamiltonian() # create the Hamiltonian
es = np.linspace(-.5,.5,50) # set of energies for dIdV
Gs = dict()
for name,m2 in [("parallel",[0.,0.,0.5]),("antiparallel",[0.,0.,-0.5])]:
    h1 = h.copy() ; h1.add_exchange([0.,0.,0.5]) # first lead, fixed magnetization
    h2 = h.copy() ; h2.add_exchange(m2) # second lead, parallel or antiparallel
    HT = heterostructures.create_leads_and_central(h1,h2,h1) # create the junction
    Gs[name] = [HT.didv(energy=e) for e in es] # calculate conductance
```

`Gs["parallel"]` and `Gs["antiparallel"]` are the conductances $G_P$ and
$G_{AP}$ at each energy, and the magnetoresistance is

$$\mathrm{MR} = \frac{G_P - G_{AP}}{G_{AP}} .$$

In the parallel configuration each spin sees the same band on both sides, so
it is transmitted perfectly, and the conductance is 2, one per spin. In the
antiparallel configuration the spin-up band of the first lead is shifted up
and the one of the second lead is shifted down. This mismatch reflects part
of the electrons, so the conductance is smaller. For this chain the effect is
a few percent, because the exchange field 0.5 is small compared with the
bandwidth 4. This is the physics of the tunneling magnetoresistance in spin
valves.

See
`jupyter-notebooks/functionalities/quantum_transport/01_metal_metal_transport.ipynb`
for an executed notebook.

## Superconductor-metal transport

We now compute the conductance between a normal metal and a superconductor,
with `heterostructures.build()` and `HT.set_coupling()`. There are two
limits. At a transparent contact, an electron with energy inside the gap
cannot enter the superconductor as a quasiparticle, so it is Andreev
reflected as a hole with probability one, and the conductance below the gap
is twice the normal one. At a tunneling contact Andreev reflection is
suppressed, because it needs two electrons to tunnel. Then the conductance
follows the density of states of the superconductor: zero inside the gap,
and coherence peaks at $\pm\Delta$.

`HT.set_coupling(T)` multiplies the hopping between the two sides of the
junction by $T$, so $T=1$ is a perfect contact and $T\ll 1$ is a tunnel
contact. The loop goes from one limit to the other:

```python
from pyqula import geometry
from pyqula import heterostructures
import numpy as np
g = geometry.chain() # create the geometry
h = g.get_hamiltonian() # create the Hamiltonian
h1 = h.copy() # first lead
h2 = h.copy() # second lead
h2.add_swave(.01) # the second lead is superconducting
es = np.linspace(-.03,.03,100) # set of energies for dIdV
for T in np.linspace(1e-3,1.0,6): # loop over transparencies
    HT = heterostructures.build(h1,h2) # create the junction
    HT.set_coupling(T) # set the coupling between the leads
    Gs = [HT.didv(energy=e) for e in es] # calculate conductance
```

`Gs` is the conductance at each energy, for the last transparency of the
loop. Compare it with its value outside the gap, $|E|>\Delta=0.01$. At
`T=1.0` the conductance inside the gap is twice the one outside. At the
smallest transparencies it goes to zero inside the gap and has peaks at the
gap edges. This is the spectroscopy of a normal STM tip on a superconductor,
and the transparency is what the experiment changes with the distance
between the tip and the sample.

See `examples/transport/andreev_reflection/main.py` for a runnable script,
and
`jupyter-notebooks/functionalities/quantum_transport/02_metal_superconductor_transport.ipynb`
for an executed notebook.

## Transport through an arbitrary finite region

In the two examples above, the scattering region is made of copies of the
unit cell of the leads. We now use any finite (0d) Hamiltonian as the
scattering region, for example a molecule or a quantum dot, with two
semi-infinite chains attached at sites `i` and `j`, with
`h.get_central_heterostructure(i,j,left=None,right=None)`:

```python
from pyqula import geometry

g = geometry.chain()
gc = g.get_supercell(5)
gc.dimensionality = 0 # a finite, 5-site cluster (no periodicity)
hc = gc.get_hamiltonian() # the central region, can be any 0d Hamiltonian

h_normal = g.get_hamiltonian() # normal lead
h_sc = g.get_hamiltonian(); h_sc.add_swave(0.05) # superconducting lead

ht = hc.get_central_heterostructure(0,4,left=h_normal,right=h_sc)
G = ht.didv(energy=0.02) # Andreev conductance, via the BdG scattering-matrix formula
```

`G` is the conductance at $E=0.02$, inside the gap $\Delta=0.05$ of the
right lead, so it comes only from Andreev reflection. The result is a normal
junction object, so `didv`, `get_dos` and `get_kappa` all work on it.

- `left`, `right`: the two leads. By default each is a spinless chain.
- `j`: the site of the second contact. By default the last site.
- At most one of `hc`, `left` and `right` can be superconducting. With two
  superconductors there is no normal lead to define the reflection, and the
  call raises a `ValueError`; use `get_dc_current` (below) instead.
- `ht.landauer()` only works for normal junctions, since the Landauer formula
  counts single electrons. With a superconductor use `ht.didv()`.

Only finite (0d) scattering regions are possible for now. See
`examples/transport/central_region_ij/main.py` for a runnable script, and
`jupyter-notebooks/functionalities/quantum_transport/03_finite_central_region_two_leads.ipynb`
for an executed notebook.

## Landauer transmission through a real-space device

There is a second way to build a junction, with `multiterminal.Device`. Here
the leads and the scattering region are given as geometries, and the
hoppings between them come from the real distances between the atoms. The
transmission is the Landauer formula above,
$T_{ij}={\rm Tr}[\Gamma_i G \Gamma_j G^\dagger]$ between leads $i$ and $j$.
This is useful for a disordered or irregular conductor, which has no unit
cell to repeat.

```python
import numpy as np
from pyqula import geometry
from pyqula import multiterminal

def lead_cell(x): # a one-site chain unit cell sitting at x
    g = geometry.chain() ; g.r = np.array([[x,0.,0.]]) ; g.r2xyz()
    return g

gc = geometry.chain().supercell(4) # the scattering region
xs = gc.r[:,0]
d = multiterminal.Device()
d.biterminal(left_g=lead_cell(min(xs)-1.),right_g=lead_cell(max(xs)+1.),
             central_g=gc,disorder=0.0)
print("ballistic transmission:",d.transmission(energy=0.0)[0])
```

A perfect chain has one open channel and nothing to scatter, so the
transmission is 1 for every energy inside the band. With `disorder` larger
than zero, random onsite energies are added in the scattering region, and
the transmission goes below 1. Its value depends on the random
configuration, so a disorder average needs several configurations.
`d.transmission(energy=e)` returns a list with one value per pair of leads,
by default only the pair `(0,1)`. `multiterminal.landauer(d,energy,ij=[(i,j)])`
gives the same with the pairs written explicitly. The same disorder physics,
with a chain of disordered cells between two clean leads, is in
`examples/transport/disordered_transport/main.py`.

## Multiple Andreev reflection and AC-Josephson current

So far all the conductances are at zero temperature and in linear response.
We now apply a finite voltage $V$ to a junction between two superconductors,
with `HT.get_dc_current()`. With a voltage, the phase difference of the two
superconductors grows in time, $\dot\phi = 2eV/\hbar$, so the problem is
periodic in time with the Josephson frequency. Two things appear. The first
is the AC Josephson effect. The second is multiple Andreev reflection (MAR):
a quasiparticle is Andreev reflected many times between the two
superconductors, and it gains an energy $eV$ at each crossing, until it can
leave above the gap. The process of order $n$ opens when $neV = 2\Delta$, so
the current shows features at $eV = 2\Delta/n$. What a measurement sees is
the time-averaged current $I_{dc}(V)$, and this is what is computed.

The voltage is moved by a gauge transformation into a time-dependent phase
of the hopping between the two leads,
$t \to t\, e^{ieVt'/\hbar}$, and the problem is solved in Floquet space, with
sidebands at energies $\epsilon + n eV$. The DC current is

$$I_{dc}(V) = \frac{e}{h}\int_0^{eV} d\epsilon\;
\sum_n {\rm Re}\,{\rm Tr}\left\{\left[G^r \Sigma_L^< + G^< \Sigma_L^a\right]_{nn}
\tau_z\right\},$$

where $G^r$ and $G^<$ are the retarded and lesser Floquet Green's functions,
$\Sigma_L^<$ and $\Sigma_L^a$ the lesser and advanced self-energies of the
left lead, and $\tau_z$ is $+1$ for electrons and $-1$ for holes. This is the
formalism of San-Jose, Cayao, Prada and Aguado,
[arXiv:1301.4408](https://arxiv.org/abs/1301.4408). The current is in units
of $e/h$ times energy (in units of $t$), so that for two normal leads
$dI_{dc}/dV$ is the conductance in units of $e^2/h$. It works for any pair of
normal and superconducting leads, also for two superconductors, where the
scattering formula of `didv` above cannot be used.

```python
from pyqula import geometry
from pyqula import heterostructures
import numpy as np
g = geometry.chain() # create the geometry
h = g.get_hamiltonian() # create the Hamiltonian
h1 = h.copy() ; h1.add_swave(0.1) # left superconducting lead
h2 = h.copy() ; h2.add_swave(0.1) # right superconducting lead
HT = heterostructures.build(h1,h2) # create the SNS junction
HT.set_coupling(0.5) # set the normal transparency of the weak link
vs = np.linspace(0.02,1.5,40)*0.1 # bias voltages
Is = HT.get_iv_curve(vs) # MAR/AC-Josephson dc current
```

`Is` is the DC current at each voltage in `vs`. Above $eV=2\Delta=0.2$ the
current grows roughly linearly with the voltage, as in a normal junction. Below
$2\Delta$ it is carried by multiple Andreev reflections, and it shows steps
at $eV = 2\Delta/n$, sharper for a small transparency.

This is one of the most expensive calculations in the guide, roughly fifteen
seconds per voltage for this junction. The reason is physical: the number of
Andreev reflections that matter is about $2\Delta/eV$, and each one needs a
Floquet sideband, so small voltages need many sidebands. The sidebands are
added until $I_{dc}$ converges. If a voltage is very small, the number of
sidebands can reach `nmax_max`, and a warning says that this point is not
converged. Starting the voltages a bit higher is the cheap solution; a larger
`nmax_max` is the expensive one.

Only 1D leads are possible. A central region between the leads, for example a
quantum dot, can be added with `heterostructures.build(h1,h2,central=[hc])`,
and it is slower. The voltage drops at the last bond on the right, so the
central region is at the potential of the **left** lead. To move its levels,
use `hc.shift_fermi(eps)`, which keeps it a valid BdG Hamiltonian, and do not
add to its diagonal by hand. See `examples/transport/floquet_keldysh_mar/main.py`
for a runnable script.

### Conductance at finite voltage

`HT.didv()` chooses between the two formalisms with `method`:

- `method="smatrix"`: the zero-temperature scattering formula above. The
  normal lead is grounded and all the voltage drops on the other lead.
- `method="keldysh"`: the derivative of $I_{dc}$ at voltage `energy`. Both
  leads are biased rigidly.
- `method="auto"` (default): `"keldysh"` if both leads are superconducting,
  and `"smatrix"` otherwise.

The two methods apply the voltage in a different way, so on a junction with
one normal lead they do not give the same number, even when the pairing of
the other lead goes to zero. Leave the default.

`didv` takes a single `energy` and returns a number, or an array
`energies=[...]` and returns an array, but not both. For the junction between
two superconductors above, this is the $dI/dV$ curve in the MAR regime:

```python
es = np.linspace(0.15,0.25,40)*0.1 # bias energies
Gs = HT.didv(energies=es, nmax_max=40) # dI/dV curve
```

### A superconducting STM tip

A `LocalProbe` is an STM tip, a semi-infinite chain coupled to one site of an
infinite sample. The tip and the sample play the role of the two leads, and
`T` is the hopping between them. The choice of method is the same as above:
the scattering formula by default, and the Floquet-Keldysh current when both
tip and sample are superconducting. For a normal tip the two methods agree,
because the tip is grounded in both.

Besides the conductance, we compute the decay rate $\kappa$. In STM, the
conductance changes with the tip height through the transparency $T$, and
the exponent of this power law depends on how many electrons are transferred
in each process. pyqula computes

$$\kappa = \frac{d\ln G_S/d\ln T}{d\ln G_N/d\ln T},$$

the ratio between the exponent with the superconductor and the exponent of
the same junction with the pairing removed. In the tunnel limit
$G_N\propto T^2$. A single quasiparticle also gives $G_S\propto T^2$, so
$\kappa=1$, while Andreev reflection transfers two electrons,
$G_S\propto T^4$, and $\kappa=2$. So $\kappa$ tells you which process carries
the current at each energy, and it is measured by changing the tip height.

```python
from pyqula import geometry
from pyqula.transporttk.localprobe import LocalProbe

g = geometry.chain()
h = g.get_hamiltonian() ; h.shift_fermi(1.) ; h.add_swave(0.1) # SC sample
lead = geometry.chain().get_hamiltonian() ; lead.shift_fermi(1.) ; lead.add_swave(0.1) # SC probe
lp = LocalProbe(h,lead=lead,delta=1e-3)
lp.T = 0.3 # reference transparency
G = lp.didv(energy=0.25,nmax=4,nmax_max=12,tol=5e-2) # routed through Keldysh automatically
k = lp.get_kappa(energy=0.25,nmax=4,nmax_max=12,tol=5e-2)
```

`G` is the conductance and `k` the decay rate at voltage $0.25$, which is
above the combined gap $2\Delta=0.2$ of tip and sample. This is much more
expensive than with a normal tip, because each `didv` or `get_kappa` call
solves several Floquet-Keldysh problems, and it is slowest deep inside the
gap at small transparency. See
`examples/transport/decay_constant_keldysh/main.py` for a runnable script
with a coarse energy grid and few sidebands, and the notebooks
`08_tunneling_contact_probe.ipynb` (normal tip) and
`09_mar_ac_josephson_sc_probe.ipynb` (superconducting tip) in
`jupyter-notebooks/functionalities/quantum_transport/`.

`get_kappa` also takes a temperature `temp`, and then every conductance is
the thermal average at that temperature. Like `didv`, it takes a single
`energy` or an array `energies=[...]`:

```python
k = lp.get_kappa(energies=[0.1,0.25,0.4],temp=0.02,nmax=4,nmax_max=12,tol=5e-2)
```

`HT.get_kappa()` of a junction takes the same `temp` and `energies`.
