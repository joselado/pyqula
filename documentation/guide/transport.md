# Quantum transport

We will now see how to perform quantum transport calculations with pyqula, starting from the
conductance of the simplest junctions between two leads and reaching the current through a
voltage-biased superconducting one.

A transport calculation puts a scattering region between two semi-infinite leads, each of them
a periodic Hamiltonian of the kind built in the previous chapters, and asks how much current
flows through it. The leads enter through their surface Green's functions, obtained with the
same decimation as in the surface spectral functions, and the quantity computed is the
differential conductance $dI/dV$, what a two-terminal measurement gives: for normal leads it
is the Landauer transmission, in units of $e^2/h$ per open channel, and when a lead is
superconducting it includes Andreev reflection, the process in which an electron arriving from
the normal side is reflected as a hole while a Cooper pair enters the superconductor.

## Magnetoresistance in metal-metal transport

As a specific example, here we will address how to compute the magnetoresistance of a junction
between two magnetic metals, with `HT.didv()`. We build two copies of the same lead, give each
one an exchange field pointing in a different direction, and compare the conductance of the
parallel and antiparallel configurations

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

The magnetoresistance follows from the two conductance curves,
$\mathrm{MR} = (G_P - G_{AP})/G_{AP}$ evaluated at the Fermi energy. In the parallel
configuration each spin channel sees the same bands on both sides of the junction and
transmits perfectly, so the conductance at zero energy is 2, one per spin; in the antiparallel
one a spin-up electron of the first lead arrives at a lead whose spin-up band is shifted the
opposite way, the mismatch reflects part of it, and the conductance is smaller. For the chain
above the effect is a few percent, since an exchange of 0.5 is small compared with the
bandwidth of 4. See
`jupyter-notebooks/functionalities/quantum_transport/01_metal_metal_transport.ipynb` for the
executed notebook.

## Superconductor-metal transport

Here we address how transport between a superconducting lead and a metallic lead is computed,
with `heterostructures.build()` and `HT.set_coupling()`. As a paradigmatic example, we will
focus on the Andreev reflection regime and the tunneling regime. Below the gap an electron from
the normal lead cannot enter the superconductor as a quasiparticle: it can only be reflected
as a hole, leaving a Cooper pair behind, and since that process transfers a charge $2e$ it
doubles the conductance of a transparent contact below the gap. At a tunneling contact Andreev
reflection is suppressed, and the conductance instead follows the density of states of the
superconductor, zero inside the gap and with coherence peaks at $\pm\Delta$. The loop over the
transparency `T` goes from one limit to the other

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

`Gs` is the conductance curve at each transparency, and comparing it against its value outside
the gap is the natural way to read it: at `T=1.0` the sub-gap conductance is twice the normal
one, and at the lowest transparencies it vanishes inside the gap and peaks at the gap edge.
This is the spectroscopy that a normal tip on a superconductor gives, and the transparency is
the knob an experiment controls through the tip-sample distance. See
`examples/transport/andreev_reflection/main.py` for a runnable version, and
`jupyter-notebooks/functionalities/quantum_transport/02_metal_superconductor_transport.ipynb`
for the executed notebook.

## Transport through an arbitrary finite region

The two examples above build the central scattering region out of copies of the leads' own
unit cell. Let us now see how any finite (0d) Hamiltonian can act as the central region
instead, contacted by two semi-infinite 1D chain leads attached at sites `i` and `j`, with
`h.get_central_heterostructure(i,j,left=None,right=None)`

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

It returns a plain `Heterostructure`, so `didv`, `get_dos`, `get_kappa` and the rest all work
on it. The exception is `landauer`, which refuses a junction with a Nambu degree of freedom:
the Landauer formula counts single-particle transmission, while a Cooper pair carries charge
$2e$. Use `didv`, which applies the BdG scattering formula, in that case. `left` and `right`
default to a plain spinless chain and `j` to the last site. At most one of `hc`, `left` and
`right` may carry a pairing amplitude, so a superconducting central region between two normal
leads is fine, and so is one superconducting lead, but two superconducting leads raise a
`ValueError`: there is then no normal lead left to define a reflection amplitude against, and
`get_dc_current`, addressed below, is the right tool instead. Only 0d central regions are
supported so far. See `examples/transport/central_region_ij/main.py` for a runnable script,
and
`jupyter-notebooks/functionalities/quantum_transport/03_finite_central_region_two_leads.ipynb`
for the executed notebook.

## Landauer transmission through a real-space device

`heterostructures` builds a junction out of two periodic Hamiltonians. Let us now see the
other route, with `multiterminal.Device`: the leads and the scattering region are given as
*geometries*, the hoppings between them are generated from the actual interatomic distances,
and the Landauer transmission follows from the leads' self-energies. It is the natural form
for a disordered or irregularly-shaped conductor, where there is no unit cell to repeat.

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

A perfectly matched chain has one open channel and nothing to scatter off, so the transmission
is 1 everywhere inside the band, to the tolerance of the lead decimation. Raising `disorder`
puts random onsite energies in the central region and the transmission drops below 1 by an
amount that depends on the realization drawn, meaning that a disorder-averaged result needs
several draws. `d.transmission(energy=e)` returns a list with one entry per lead pair, the
pair `(0,1)` by default; `multiterminal.landauer(d,energy,ij=[(i,j)])` is the same quantity
as a plain function, with the pairs named explicitly. The same disorder physics done with the
junction machinery, a chain of disordered cells between two clean leads, is in
`examples/transport/disordered_transport/main.py`.

## Multiple Andreev reflection and AC-Josephson current

`didv` and `landauer` above are equilibrium, zero-bias linear-response quantities. Let us now
see what happens when a finite voltage is applied across a junction between two
superconductors, an SNS junction, with `HT.get_dc_current()`. A finite bias makes each lead's
pairing phase wind in time, giving rise to multiple Andreev reflections (MAR), in which a
quasiparticle bounces between the two superconductors gaining $eV$ at each crossing until it
reaches the gap edge, and to an AC Josephson effect; the physically meaningful, measurable
quantity is the time-averaged (DC) current $I_{dc}(V)$. `Heterostructure.get_dc_current(voltage)`
computes it with the Floquet-Keldysh formalism of San-Jose, Cayao, Prada and Aguado, *New J.
Phys.* **15**, 075019 ([arXiv:1301.4408](https://arxiv.org/abs/1301.4408)): the bias is
gauged away from the (static) leads into a single, periodically time-dependent "weak link"
hopping, and the resulting Floquet-space Dyson/Keldysh equations are solved to get
$I_{dc}(V)$. It works for any combination of normal and superconducting leads, including the
case of **two** superconducting leads, which the scattering-matrix formula behind `didv`
cannot handle on its own (it has no normal lead to define a reflection amplitude against).

`didv` reaches the same formalism through a `method` argument, so a conductance can be had
without differentiating $I_{dc}$ by hand. `method="smatrix"` is the zero-temperature BdG
scattering-matrix conductance, `method="keldysh"` differentiates `get_dc_current` at the bias
`energy`, and the default `method="auto"` picks Keldysh when both leads are genuinely
superconducting and the scattering matrix otherwise.

The two are different physical quantities rather than two ways of getting the same one,
because the bias is applied differently: `"smatrix"` grounds the normal lead and drops the
whole bias on the other, while `"keldysh"` biases both leads rigidly. So on a junction with one
normal lead they disagree by an O(1) factor, and they keep disagreeing as the other lead's
pairing is taken to zero. Let `method="auto"` choose.

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

`Is` is the dc current at each bias, the curve a current-biased measurement of an SNS junction
traces out. This is among the most expensive calculations in the guide, at roughly fifteen
seconds per bias point for the SNS junction above, so a forty-point curve takes minutes. The
reason is physical: the number of Andreev reflections that matter is the MAR order
$2\Delta/eV$, which grows as the bias falls, and each one costs a Floquet sideband. The
sidebands are added automatically until $I_{dc}$ converges, but a grid reaching very close to
zero bias can exhaust `nmax_max` and warn that the sidebands did not converge. That point is
inaccurate; nothing has crashed. Starting the grid a little higher is the cheap fix; raising
`nmax_max` the expensive one.

Only 1D leads are supported. A central region between the leads
(`heterostructures.build(h1,h2,central=[hc])`, for example a quantum dot) works too, more
slowly. The bias drops across the junction's rightmost bond, so the central region sits at the
**left** lead's potential; detune it with `hc.shift_fermi(eps)`, which keeps it a valid BdG
Hamiltonian, rather than by adding to its diagonal. See
`examples/transport/floquet_keldysh_mar/main.py` for a runnable script.

A `LocalProbe` is an STM-like tip weakly coupled to one site of an infinite sample, with the
tip and the sample site playing the role of the two leads. The same choice of method applies:
the scattering matrix by default, and the Floquet-Keldysh MAR current when tip and sample are
both superconducting, which is the situation of a superconducting tip on a superconducting
sample. On a `LocalProbe` the two methods do agree when the tip is normal, since the tip is
grounded there exactly as the scattering matrix grounds it.

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

This is considerably more expensive than the normal-probe case (each `didv` or `get_kappa`
call runs several Floquet-Keldysh sideband sweeps), especially deep below the combined gap at
low transparency, where the sideband sum converges slowly; see
`examples/transport/decay_constant_keldysh/main.py` for a runnable script using a coarse
energy grid and a modest sideband cutoff to keep the runtime reasonable, and
`jupyter-notebooks/functionalities/quantum_transport/08_tunneling_contact_probe.ipynb` and
`09_mar_ac_josephson_sc_probe.ipynb` in the same folder for the executed notebooks on the
normal and the superconducting probe.

`get_kappa` also accepts a `temp` argument for a thermally-averaged kappa (each conductance
entering the power-law fit is `didv(temp=...)`'s thermal average rather than the
zero-temperature value). Pass a single `energy` (returns a scalar, as above) or a whole
`energies=[...]` array at once (returns an array):

```python
k = lp.get_kappa(energies=[0.1,0.25,0.4],temp=0.02,nmax=4,nmax_max=12,tol=5e-2)
```

`Heterostructure.get_kappa` takes the same `temp` and `energies` arguments.

`didv`'s `energy` and `energies` arguments are mutually exclusive, the same convention
`get_kappa` uses: pass a single `energy` and get a scalar back, or a whole `energies=[...]`
array at once and get an array.

```python
es = np.linspace(0.15,0.25,40)*0.1 # bias energies
Gs = HT.didv(energies=es, nmax_max=40) # dI/dV curve
```
