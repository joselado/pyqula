# Charge and generic response functions

## Charge-charge response function

Let us now look at the simplest response function, the one of the density at
site $i$ to a potential at site $j$, with `h.get_chi()`. The charge-charge
response function for a spinless system is computed as
$$
\chi(\omega,i,j) = 
\sum_{n,m}
\left[f(\epsilon_n) - f(\epsilon_m)\right]
\frac{
\Psi^*_n(i)\Psi_m(i)
\Psi^*_m(j)\Psi_n(j)
}
{
\epsilon_n - \epsilon_m - \omega + i\delta
}
$$

where $f(\epsilon)$ is the Fermi-Dirac distribution, so that a pair of states
contributes only when their occupations differ, weighted by the amplitudes of
both states at the two sites and resonating at their energy difference. A
transition from an occupied state $m$ to an empty state $n$ resonates at
$\omega=\epsilon_n-\epsilon_m>0$, so the imaginary part of the diagonal
response $\chi(\omega,i,i)$ is positive at positive frequency and negative at
negative frequency. For a periodic system the response is computed at a
momentum transfer $q$, summing over the k-mesh

```python
from pyqula import geometry
g = geometry.chain() # create honeycomb lattice
h = g.get_hamiltonian() # create hamiltonian of the system
(es,chis) = h.get_chi(q=[0.,0.,0.]) # get response function
```

`es` holds the frequencies and `chis` one complex number per frequency, the
response traced over the sites of the unit cell, whose imaginary part is what
inelastic scattering measures. Note that at $q=0$ the response of the chain
vanishes at any finite frequency, since with a single band there is no state
to scatter into at the same momentum, and only its static limit survives,
$\chi(\omega\to0)=-N(0)$ with $N(0)=1/\pi t$ the density of states at the
Fermi level counting both spins, $-0.32$ here, spread by the broadening into
a peak of width $\delta$ around zero frequency; the scan below, at finite $q$,
is where the particle-hole continuum appears. That static limit comes from
the pairs of states with the same energy, which on the k-mesh stand for the
transitions within a band around them and enter with the derivative of the
Fermi function, the $\delta\to0$ limit of the occupation factor over the
energy difference. When $\delta$ is instead larger than the spacing of the
energy differences on the mesh, the neighboring k-points already resolve
those transitions and such a pair adds an error of order $1/n_k$, so a
converged calculation at finite $\delta$ is better done with an odd `nk`,
which keeps the pairs at a nesting wavevector off the mesh.

Optional arguments
- q: momentum transfer of the response function
- energies: array of frequencies
- delta: broadening
- nk: number of k-points used in the Brillouin-zone integration

The charge-charge response can also be scanned over momentum transfer to build a $\chi(q,\omega)$ map

```python
import numpy as np
g = geometry.chain()
h = g.get_hamiltonian()
qs = np.linspace(-1.,1.,40)
chis = [h.get_chi(q=[q,0.,0.],energies=np.linspace(-3,3,100),nk=40,delta=0.1)[1] for q in qs]
```

What you see in that map is the particle-hole continuum of the band, the
energies a scattering from $k$ to $k+q$ across the Fermi level can cost. See
`examples/1d/charge_response/main.py` for a runnable version, and
`jupyter-notebooks/functionalities/response_functions/03_charge_response.ipynb`
for an executed notebook.

## Generic operator-operator response function

The same machinery works for any pair of operators. `h.get_chi()` computes
the charge-charge response by default, but any two operators $A$ and $B$ can
be used instead, giving the generalized response

$$
\chi_{AB}(\omega,q) = \sum_{k,n,m}
\left[f(\epsilon_{k,n}) - f(\epsilon_{k+q,m})\right]
\frac{
\langle \Psi_{k,n}|A|\Psi_{k+q,m}\rangle
\langle \Psi_{k+q,m}|B|\Psi_{k,n}\rangle
}
{
\epsilon_{k,n} - \epsilon_{k+q,m} - \omega + i\delta
}
$$

where the perturbation couples through $B$, the measurement is of $A$, and
the sum runs over the pairs of states at $k$ and $k+q$ whose occupations
differ.
Taking both operators to be the spin along $z$ gives the longitudinal spin
susceptibility of the chain

```python
from pyqula import geometry
import numpy as np
g = geometry.chain()
h = g.get_hamiltonian(has_spin=True)
sz = h.get_operator("sz") # any of the operators from the "Operators" section
(es,chi) = h.get_chi(A=sz,B=sz,energies=np.linspace(-2.,2.,100),nk=40,delta=0.1)
```

By default `A=B=`identity, which recovers the charge-charge response above.
Any operator from the "Operators" section (spin, valley, sublattice, location,
Nambu...) can be plugged in to build the corresponding susceptibility. See
`jupyter-notebooks/functionalities/response_functions/04_operator_response.ipynb`
for an executed notebook.

## RKKY response function

Let us now use that response to couple two magnetic impurities. A moment
placed in a metal polarizes the conduction electrons around it, and a second
moment at some distance feels that polarization: the result is an effective
exchange coupling between the two, the RKKY (Ruderman-Kittel-Kasuya-Yosida)
interaction, which follows from the same non-interacting response function.
`rkky.rkky_map` computes it between a reference site and every other site in
the system, as a function of distance

```python
from pyqula import geometry
from pyqula import rkky
g = geometry.chain()
h = g.get_hamiltonian()
h.add_onsite(1.)
m = rkky.rkky_map(h,n=10,mode="LR",nk=200) # linear-response RKKY vs distance
```

Optional arguments
- mode: `"LR"` for linear response, the same response function `get_chi` uses, or `"pm"` ("poor man's"), which adds a small magnetic perturbation at each site and measures the change in total energy
- n: how many neighboring cells/distances to compute
- nk: number of k-points used in the Brillouin-zone integration

`m` is an array whose columns are `(distance, ..., ..., RKKY energy)`; the RKKY energy is in the last column

```python
x,e = m[:,0],m[:,3]
```

What you see is a coupling that alternates in sign with the distance and
decays, with a period set by the Fermi wavevector, which is why the snippet
shifts the Fermi energy with the onsite term first.

See `examples/1d/RKKY/main.py` and `examples/1d/rkky_minimal/main.py` (which compares `"pm"` and `"LR"` on the same system) for runnable versions, and `jupyter-notebooks/functionalities/response_functions/05_rkky.ipynb` for an executed notebook.
