# Pseudofermion mean field

In this page we treat models of local spins with the same mean-field method
as electrons. Each spin one half is written with an auxiliary fermion, the
Abrikosov pseudofermion,

$$
\vec S_i=\tfrac12 \sum_{ss'} f^\dagger_{i,s}\vec\sigma_{ss'} f_{i,s'},
\qquad
\sum_s f^\dagger_{i,s} f_{i,s}=1 ,
$$

with exactly one fermion on each site, because a spin one half has no charge
fluctuations. You can think of the pseudofermion as the spin part of an
electron, with the charge frozen. We use it first for the Heisenberg model,
where it gives spinons, and then for the Kondo lattice, where it gives heavy
fermions. For both we also compute what a tunneling microscope measures. For the
Kondo lattice we also compute the spectral function of the composite fermion
and the spin susceptibility of the moments.

## Spinon mean field for Heisenberg models

We take a spin-$\tfrac12$ Heisenberg model,

$$
H=J\sum_{\langle ij\rangle}\vec S_i\cdot\vec S_j ,
$$

where no electron hops. With the pseudofermions, the exchange term is quartic
in $f$. Its mean-field decoupling gives a bond order parameter

$$
\chi_{ij}=\langle f^\dagger_i f_j\rangle ,
$$

which acts as a hopping for the pseudofermions, so they get a dispersion.
These dispersing fermions are the spinons, and this is the
resonating-valence-bond ansatz. The decoupling is the same one that the $J$
channel of `h.get_combined_mean_field_hamiltonian()` does (see
[Exchange interactions](exchange.md)), with two differences: the bare hopping
is zero, and the constraint of one fermion is imposed on every site, not only
on average. Each site gets its own Lagrange multiplier $\lambda_i$, a local
chemical potential, chosen so that $\langle n_i\rangle = 1$. The construction
follows section 4 of arXiv:1601.03742. `SpinonHamiltonian` (in
`pyqula.spinon`) builds it:

```python
from pyqula import geometry
from pyqula.spinon import SpinonHamiltonian

g = geometry.triangular_lattice() # a canonical frustrated-Heisenberg lattice
h = SpinonHamiltonian(g) # zero bare hopping, couplings come from J1/J2/...
h2 = h.get_mean_field_hamiltonian(J1=1.0,nk=12,mix=0.1,maxerror=1e-4)

h2.local_occupation   # <n_i> per site, exactly 1.0 at convergence
h2.constraint_lambda  # converged per-site Lagrange multiplier (local chemical potential)
h2.get_bands()        # spinon dispersion
```

The result is a spinon Hamiltonian:

- `local_occupation`: the occupation of each site, equal to one at
  convergence.
- `constraint_lambda`: the Lagrange multiplier $\lambda_i$ of each site.
- `h2.get_bands()`: the spinon dispersion. In a quantum spin liquid these
  spin-$\tfrac12$ excitations replace the magnons of an ordered magnet, and
  they appear as a continuum in inelastic neutron scattering.

The keyword `filling` is not accepted here, because the representation is only
valid with exactly one fermion per site. Only the U(1) ansatz, with the bond
order $\chi_{ij}$ alone, is implemented. The Z2 ansatz, which also includes
the pairing of spinons that $J$ can induce, is not.

On a frustrated lattice (triangular, kagome) the converged state is not
unique. Several self-consistent states, with different fluxes of $\chi_{ij}$
around a plaquette, can exist at the same $J$, and a random guess can land on
any of them. This is part of the physics of the mean-field ansatz, not a
problem of the loop. On a chain with one site per unit cell there is no
frustration and the solution is unique, so repeated runs agree within
`maxerror`. On a frustrated lattice, give an explicit `mf` to choose the
ansatz you want.

A Zeeman field is already bilinear in $f$, so it enters exactly, without any
decoupling. We add it with `h.add_zeeman()` or `h.add_exchange()` on the
`SpinonHamiltonian`, before the mean-field calculation:

```python
h = SpinonHamiltonian(g)
h.add_zeeman([0., 0., 0.3])            # or h.add_exchange([0.,0.,0.3])
h2 = h.get_mean_field_hamiltonian(J1=1.0, nk=12)
h2.get_magnetization()                 # induced <sigma> per site, twice <S>
```

The vector $\vec b$ given to `h.add_zeeman()` multiplies the Pauli matrices,
not $\vec S=\vec\sigma/2$. The term is $+\vec b\cdot\vec\sigma=2\vec
b\cdot\vec S_i$, so the induced $\langle\vec S_i\rangle$ points opposite to
$\vec b$. This is the same convention as for electrons in
[Building a Hamiltonian](hamiltonian.md). The constraint fixes the total
occupation, not the spin, so it stays exact in a field, and
$\langle \vec S_i\rangle$ grows with the field until it saturates when the
field is large compared with $J$.

In a field the loop is fragile. With the snippet above, only 1 of 5 random
seeds converges, and the converged state has a different bond order
$\chi_{ij}$ from the state without field, so its spinon band has another
shape.

![Spinon bands of the triangular lattice without and with a Zeeman field, colored by the spin of each state.](img/pseudofermions_spinon.png#only-light)
![Spinon bands of the triangular lattice without and with a Zeeman field, colored by the spin of each state.](img/pseudofermions_spinon-dark.png#only-dark)

The script fixes the random seed, so that the figure is reproducible. On the left, the
spinon band without field goes from $-0.47$ to $0.57$ in units of $J$. It is
half filled, so the spinons have a Fermi surface. On the right, with
$b_z=0.3$, the two spin bands are split by $0.1$ and
$\langle\sigma_z\rangle=-1/6$, so $\langle S_z\rangle=-1/12$ per site, against
the field vector. The splitting is much smaller than the bare $2b_z=0.6$. The
reason is the exchange with the six neighbors, which adds a field
$6J\langle S_z\rangle=-0.5$ on $S_z$ and cancels most of it. The band also
changes its shape, because the bond order in the field is another one.

??? example "Code of this figure (documentation/figures/pseudofermions_spinon.py)"

    ```python
    --8<-- "pseudofermions_spinon.py"
    ```

See `examples/spinon/heisenberg_chain/main.py` and
`examples/spinon/heisenberg_chain_zeeman_field/main.py` for runnable scripts,
and the notebook
`jupyter-notebooks/functionalities/interacting_mean_field_hamiltonians/13_spinon_rvb.ipynb`.

## Spin liquid ansatz and its spin response

We will now see how to keep the mean field in a chosen spin liquid, and how
to compute its dynamical spin response. The mean field of the previous
section has a magnetic channel. When the unit cell allows a magnetic order,
the loop usually finds it, and the spin liquid is lost. To study the liquid
itself we keep only the bond channel. With one fermion per site the exchange
is exactly

$$
\vec S_i\cdot\vec S_j=-\tfrac12\,\hat\chi^\dagger_{ij}\hat\chi_{ij}+\tfrac14 ,
\qquad
\hat\chi_{ij}=\sum_s f^\dagger_{i,s}f_{j,s} ,
$$

and its decoupling gives spinons that hop between the sites with the
amplitude

$$
|t_{ij}|=\tfrac12 J_{ij}\,|\chi_{ij}| ,\qquad \chi_{ij}=\langle\hat\chi_{ij}\rangle ,
$$

meaning that there is no magnetic mean field at all. This is the mean field
of arXiv:2503.13831, and we select it with `decoupling="bond"`. In the first
neighbor chain it gives $|t|=J/\pi$, where the default decoupling gives
$3J/(4\pi)$.

The bond mean field alone does not fix the spin liquid either. A free loop
goes to the state of lowest mean-field energy, which is often not the one we
want: in the chain with two sites per cell it is a dimerized state, and in
the triangular lattice it is a state that breaks time reversal. So we give
the hopping pattern of the liquid with `ansatz`. Every hopping keeps the
phase that it has in the ansatz, a hopping that is zero in the ansatz stays
zero, and the loop converges the amplitudes. The amplitudes are free. When
all the bonds of the pattern are equivalent they stay equal, as below. In a
cell with inequivalent sites or bonds the loop can still go to a dimerized
state. With `hold="pattern"` the relative amplitudes are held too. The
hoppings are then

$$
t_{ij}=c\,J_{ij}\,P_{ij} ,
$$

with $P_{ij}$ the ansatz, and the loop converges only the amplitude $c$ and
the Lagrange multiplier of each site. This is the choice for a large cell
with inequivalent sites.

As an example we take the $J_1$-$J_2$ Heisenberg model in the triangular
lattice, and the $\pi$-flux pattern, which has flux $\pi$ in the up triangles
and zero in the down ones. Its spinons have Dirac cones, so this is a Dirac
spin liquid. The pattern needs two sites per cell:

```python
import numpy as np
from pyqula import geometry
from pyqula.spinon import SpinonHamiltonian

g0 = geometry.triangular_lattice()
g = g0.get_supercell((2,1)) # two sites, the cell of the pi-flux pattern
a = g.get_hamiltonian(has_spin=True) # first-neighbor hoppings, all +1
ansatz = dict()
for (d,m) in a.get_dict().items(): # put the signs of the pi-flux pattern
    m = np.array(m)
    for i in range(2):
        for j in range(2):
            dr = g.r[j] + d[0]*g.a1 + d[1]*g.a2 - g.r[i] # the bond
            if abs(dr[1])<1e-6: continue # a bond along a1 keeps +1
            p = i if dr[1]>0 else i + int(round(dr[0]-0.5)) # its lower end
            m[2*i:2*i+2,2*j:2*j+2] *= (-1)**p
    ansatz[d] = m

h = SpinonHamiltonian(g)
h2 = h.get_mean_field_hamiltonian(J1=1.0,J2=0.125,decoupling="bond",
        ansatz=ansatz,gutzwiller=2.0,nk=20,T=0.02)
```

The converged hoppings are real and all have the same modulus, $0.395J$. The
second-neighbor bonds are zero in the ansatz, so $J_2$ does not change the
spinon bands. It enters in the spin response below. The keywords are:

- `decoupling`: `"hartree-fock"` (the default, with the magnetic channel) or
  `"bond"`. The bond one takes an isotropic exchange.
- `ansatz`: the hopping pattern to hold, a Hamiltonian or a dictionary of
  hopping matrices. Without it the pattern can change during the loop.
- `hold`: `"phases"` (the default) holds the phases of the ansatz and leaves
  the amplitudes free. `"pattern"` holds the relative amplitudes too.
- `gutzwiller`: a factor that multiplies the exchange everywhere. Without the
  constraint the local moment is $\langle\vec S_i^2\rangle=3/8$ instead of
  $3/4$, and a factor of 2 accounts for this in an approximate way. It
  doubles the energies and does not change the stability.
- `T`: the temperature of the loop. For Dirac spinons a small temperature,
  as here, keeps the two sites equivalent on a k-mesh.

The spinons of the mean field do not interact. The exchange between them
comes back in the spin response, in the random phase approximation,

$$
\chi(\vec q,\omega)=\chi_0(\vec q,\omega)
\left[1+J(\vec q)\,\chi_0(\vec q,\omega)\right]^{-1} ,
$$

where $\chi_0$ is the response of the free spinons, the two-spinon continuum,
and $J(\vec q)$ is the Fourier transform of the exchange. Both are matrices
in the sites of the cell. `h2.get_spinchi_full()` returns $\chi$, and
$\chi_0$ with `RPA=False`:

```python
energies = np.linspace(0.,3.,100)
K = [2/3,1/3,0.] # the K point of the lattice, in the vectors of the supercell
(es,chi0) = h2.get_spinchi_full(q=K,energies=energies,delta=0.05,nk=31,RPA=False)
(es,chi) = h2.get_spinchi_full(q=K,energies=energies,delta=0.05,nk=31)
r = h2.get_rpa_stiffness(K,nk=31,delta=1e-3,T=2e-3)
```

Each `chi[n]` is a matrix between $S_x$, $S_y$ and $S_z$ of every site, as in
[Spin response and magnons](spin-response.md). Its imaginary part is what
inelastic neutron scattering measures. `h2.get_rpa_stiffness(q)` is the
smallest eigenvalue of the static kernel $1+J(\vec q)\chi_0(\vec q,0)$. It is
1 without exchange, and it goes to zero when the liquid becomes unstable to
magnetic order with wavevector $\vec q$. A negative value means that the
liquid is not a stable state of the model. Here it is $0.12$ at K. Use an odd
`nk` in the response, so that the Dirac points are not on the k-mesh.

![Imaginary part of the spin response of the Dirac spin liquid of the triangular lattice along Gamma, K, M, Gamma, for free spinons and in the random phase approximation.](img/pseudofermions_spinliquid.png#only-light)
![Imaginary part of the spin response of the Dirac spin liquid of the triangular lattice along Gamma, K, M, Gamma, for free spinons and in the random phase approximation.](img/pseudofermions_spinliquid-dark.png#only-dark)

On the left, the free spinons give only a continuum, with most of its weight
above $1.5J$. On the right, the exchange pulls the weight down into a sharp
mode below the continuum, a bound state of two spinons with spin one. Its
energy is $0.64J$ at K. It gets softer towards M, where the continuum reaches
zero energy because of the Dirac cones. When $J_2$ decreases the mode at K
goes down, and it reaches zero at $J_2=0.0716J_1$, the transition to the
$120^\circ$ order. At M it reaches zero at $J_2=0.176J_1$, the transition to
the stripe order. Between the two values the liquid is stable in this
approximation. The figure shows the response at each momentum of the
lattice, which is a plane wave on the two sites of the cell.

??? example "Code of this figure (documentation/figures/pseudofermions_spinliquid.py)"

    ```python
    --8<-- "pseudofermions_spinliquid.py"
    ```

A Zeeman field enters as in the previous section, with `h.add_zeeman()`
before the mean-field calculation. The loop then adds the field of the
neighboring moments, the Weiss field
$\sum_j J_{ij}\langle\vec S_j\rangle$ acting on $\vec S_i$, so that the
induced moment feels the exchange:

```python
h = SpinonHamiltonian(g)
h.add_zeeman([0.,0.,0.1])
h2 = h.get_mean_field_hamiltonian(J1=1.0,J2=0.125,decoupling="bond",
        ansatz=ansatz,nk=20,T=0.02)
m = h2.get_magnetization() # induced <sigma> per site
(es,chi) = h2.get_spinchi_full(q=[0.,0.,0.],energies=np.linspace(0.,0.6,121),
        delta=0.01,nk=31,T=0.02)
```

The induced moment is $\langle\sigma_z\rangle=-0.010$ on each site. Without
the Weiss field (`weiss=False`) it would be $-0.013$, so the antiferromagnetic
exchange reduces it. The uniform transverse response, the $S_x$ elements of
`chi` summed over the two sites, has its peak at $0.2$. This is the Zeeman
energy $2b_z$, as it must be: the total spin commutes with the exchange, so
the exchange cannot move this mode. For this the response needs the
temperature of the loop, `T=0.02`. Without the Weiss field the peak is at
$0.24$. The keyword `weiss` is `True` by default when there is a field and
`False` when there is none, so that without a field the loop cannot leave
the liquid for a magnetic order. If the stiffness is negative, a field with
the Weiss field drives the loop to a magnetic order, so check the stiffness
first. In a field the response along the field is
only approximate: a perturbation along the moment also changes the
hoppings, and the approximation does not include this. The transverse
response does not have this problem.

The random phase approximation is uncontrolled, like the mean field. It keeps
the constraint of one fermion per site only on average, so the free spinons
have half of the total weight of a spin one half. On a lattice with nesting,
like the antiferromagnetic chain at $q=\pi$, the kernel is negative at low
temperature and the approximation cannot be used. `h2.get_magnon_bands()` gives
the energy of the sharp mode along a q-path. See
`examples/spinon/dirac_spin_liquid_rpa/main.py` for a runnable script, and
`examples/spinon/spin_liquid_large_cell/main.py` for a cell of 36
inequivalent sites with `hold="pattern"` and a Zeeman field.

## Tunneling spectrum of the spinons

We will now see what a tunneling microscope measures above one site of the
spin model, when the spins sit on a metallic substrate. The electron that
tunnels can flip the spin of the site, and it pays the energy of the spin
excitation with the bias. We follow the perturbation theory of
arXiv:1505.04430 in the exchange between the tunneling electron and the
spin. To second order the conductance is

$$
\frac{dI}{dV}(eV)\propto\int d\omega\, S_i(\omega)
\left[\Theta(eV-\omega)+\Theta(-eV-\omega)\right],
\qquad
S_i(\omega)=\sum_a\int\frac{dt}{2\pi}e^{i\omega t}
\langle S^a_i(t)S^a_i(0)\rangle ,
$$

where $S_i(\omega)$ is the local spin spectrum of site $i$ and $\Theta$ is a
step broadened by the temperature. So each spin excitation of energy $\omega$
opens a step at $eV=\pm\omega$, and $dI/dV$ is the integral of the spin
spectrum up to the bias. This is inelastic tunneling spectroscopy. For
spinons a spin flip moves a spinon from an occupied state to an empty one,
so $S_i(\omega)$ is the convolution of the occupied and the empty local
density of states of the site.

To third order the electron also scatters with the electrons of the
substrate before or after the spin flip. This adds to each step a logarithm,

$$
J\rho_s\,\ln\frac{\omega_0}{|eV-\omega|} ,
$$

with $J\rho_s$ the dimensionless exchange with the substrate and $\omega_0$
its band cutoff. If the spin can flip at no cost, $\omega=0$, the logarithm
is at zero bias. This is what is called the Kondo peak. With spinons the
three spin operators of this term are again products of the local density
of states, so the method only needs the spinon states on the site of the tip.
`h2.get_kondo_spectrum()` computes both orders for a converged spinon
Hamiltonian `h2`.

The simplest case is one site without exchange. Its spinon level is at the
Fermi energy and holds one fermion, so this is a free spin:

```python
import numpy as np
from pyqula import geometry
from pyqula.spinon import SpinonHamiltonian
g = geometry.chain()
g.dimensionality = 0 # one isolated site
h1 = SpinonHamiltonian(g)
kT = 8.617333262e-2 # 1 K in meV
(es,didv) = h1.get_kondo_spectrum(np.array([0.,4.]),site=0,Jrho_s=-0.05,T=kT,
        omega0=20.,Gamma0=5e-3)
```

Here the energies are in meV, with the parameters of the spin one half of
arXiv:1505.04430. `didv` is in units of $e^2T_0^2/\hbar$, with $T_0$ the
tunneling strength. Twice `didv` divided by $2\pi$ is $1.137$ at zero bias
and $0.884$ at 4 mV, and the paper has $1.13$ and $0.886$. The factor of two
is the price of the mean field: the free fermion can also be empty or doubly
occupied, these two states carry no spin, and the local moment is
$\langle S^2\rangle=3/8$ instead of $3/4$. Without the Kondo term
(`order=2`) the spectrum is flat, $0.75$ in the same units.

We now take two chains. The first one is the uniform Heisenberg chain. The
second one is a chain of trimers, with exchange $1$ inside a trimer and
$0.1$ between two trimers:

```python
energies = np.linspace(-1.5,1.5,301)
h = SpinonHamiltonian(geometry.chain())
h2 = h.get_mean_field_hamiltonian(J1=1.0,decoupling="bond",nk=60,T=0.01)
(es,dchain) = h2.get_kondo_spectrum(energies,site=0,Jrho_s=-0.05,T=0.005,
        omega0=2.0,nk=400)

g = geometry.chain().get_supercell(3) # the sites are at x = -1, 0, 1
def J(r1,r2):
    if abs(np.linalg.norm(r1-r2)-1.)>1e-3: return 0. # first neighbors only
    inside = abs((r1[0]+r2[0])/2.)<1. # a bond inside the cell
    return 1.0 if inside else 0.1
h = SpinonHamiltonian(g)
h2 = h.get_mean_field_hamiltonian(Jr=J,rcut=1.5,decoupling="bond",
        ansatz=g.get_hamiltonian(has_spin=True),hold="phases",nk=60,T=0.01)
(es,dtrimer) = h2.get_kondo_spectrum(energies,Jrho_s=-0.05,T=0.005,
        omega0=2.0,nk=400)
```

`dchain` is one array, and `dtrimer` has one row for each site of the cell.
In the uniform chain the spinon band crosses the Fermi energy, so the spin
spectrum grows linearly from zero energy. The conductance then starts as the
square of the bias: it is $0.0005$ at zero bias and $0.015$ at $eV=0.1J$.
There is no spin flip at zero energy, and the third order adds no peak. You
can think of it this way: in the mean field the other spinons already screen
the spin of the site.

The trimers are different. A trimer has three spinon levels and three
spinons, so its middle level, at the Fermi energy, holds one spinon. This
spinon can flip at no cost. On the two end sites `dtrimer` has a maximum at
zero bias, $0.805$ against $0.727$ at $eV=0.1J$, and the second order alone
is flat there, $0.589$. On the middle site the spectrum is zero at low bias,
because the level at the Fermi energy has no weight on it. So a zero-bias
peak in this approximation says that the site has a spinon level at the
Fermi energy, a free moment, and a continuum of spinons gives none. At a
bias above the spinon bandwidth every spin flip is open and all the sites
reach $2\pi\cdot 3/8$.

Note that the level has to be sharper than the temperature. Here the
mean-field loop decouples the trimers completely: the hopping between two
of them converges to $2\times10^{-4}$, so these are isolated trimers. With
an exchange of $0.3$ between the trimers the hopping is $0.054$, the level
becomes a narrow band, and the end sites have no maximum at zero bias:
$0.018$ there against $0.64$ at $eV=0.1J$.

The keywords are:

- `site`: the site under the tip, or a list of sites. By default every site
  of the cell, one row each.
- `Jrho_s`: the dimensionless exchange with the substrate, negative for an
  antiferromagnetic one, as in the paper.
- `order`: 2 for the steps alone, 3 (default) to add the Kondo term.
- `T`: the temperature of the tip and the substrate, in the units of the
  exchange. `Ts` is the temperature of the spinon occupations, equal to `T`
  by default; use the temperature of the mean-field loop.
- `omega0`, `Gamma0`: the band cutoff and the lifetime broadening of the
  logarithm.
- `nk`: the k-mesh. The spacing of the spinon levels must be smaller than `T`.
- `de`: the step of the energy grid, `T/4` by default.

A Zeeman field, added with `h.add_zeeman()` before the mean-field
calculation, splits the peak, as it does for a spin. This is perturbation
theory in $J\rho_s$, on top of the mean field: it does not describe the
screening of the moment below the Kondo temperature, and the Gutzwiller
projection is not included. See `examples/spinon/kondo_spectrum/main.py` for
a runnable script.

## Kondo lattice

The Kondo lattice has a local moment on each site, coupled by exchange to a
conduction electron on the same site. It is the minimal model of heavy-fermion
compounds. We write it in the Coqblin-Schrieffer form of
arXiv:cond-mat/0612006 (section II.C),

$$
H=\sum_k\epsilon_k c^\dagger_kc_k + \frac{J}{N}\sum_j \sum_{ab}
S_{ab}(j)\,c^\dagger_{jb}c_{ja},
\qquad
S_{ab}(j) = f^\dagger_{ja}f_{jb} - \frac{n_f}{N}\delta_{ab},
$$

with $N=2$ for a spin one half, $a,b$ the spin indices, and $\epsilon_k$ the
conduction band. Note that $J$ here is the Coqblin-Schrieffer coupling, which
enters as $J/N$; it is not the coefficient of a Kondo term written as
$J\vec S_j\cdot\vec s_j$, and the two differ by a numerical factor. The moments
are written with pseudofermions as before, and the exchange is decoupled into
a hybridization field

$$
V_j=-\frac{J}{2}\sum_a\langle c^\dagger_{ja}f_{ja}\rangle ,
$$

which enters the mean-field Hamiltonian as $V_jf^\dagger_jc_j+\mathrm{h.c.}$,
together with a Lagrange multiplier $\lambda_j$ that keeps one fermion per
f-site. You can think of $V_j$ as the amplitude for a conduction electron to
turn into a spin flip of the local moment and back. This is the Read-Newns
(large-$N$) mean field of the Kondo lattice.

`KondoLatticeHamiltonian` (in `pyqula.kondolattice`) takes a
conduction-electron Hamiltonian and adds one localized f-site per conduction
site, with zero bare hopping. Then it solves for $V_j$ and $\lambda_j$
self-consistently:

```python
from pyqula import geometry
from pyqula.kondolattice import KondoLatticeHamiltonian

gc = geometry.chain()
hc = gc.get_hamiltonian(has_spin=True) # conduction electrons
h = KondoLatticeHamiltonian(hc)

seed = ([0.3+0.0j],[0.0]) # (V,lam), see the caveat below for why
h2 = h.get_mean_field_hamiltonian(J=1.5,filling=0.15,nk=200,mf=seed)

h2.local_occupation   # <n_f> per localized site, exactly 1.0 at convergence
h2.hybridization      # converged V per localized site
h2.constraint_lambda  # converged per-site Lagrange multiplier
```

The result is the heavy-fermion band structure. The conduction band
hybridizes with a flat f-level at an energy set by $\lambda_j$, and $V_j$
opens a hybridization gap there. The bands on both sides of the gap are flat,
heavy, because most of their weight is on the f-sites. This is the
heavy-fermion band seen in ARPES, and the hybridization gap is seen in STM
and optical spectroscopy. The three attributes are the occupation of each
f-site, the hybridization $V_j$ and the multiplier $\lambda_j$.

![Heavy-fermion bands of the Kondo chain, colored by the weight of each state on the conduction and on the f-sites.](img/pseudofermions_kondo.png#only-light)
![Heavy-fermion bands of the Kondo chain, colored by the weight of each state on the conduction and on the f-sites.](img/pseudofermions_kondo-dark.png#only-dark)

Energies are measured from the chemical potential (dashed line). Blue states
live on the conduction sites and red ones on the f-sites. The gray line is
the bare conduction band, $2t\cos k$, as a reference. The f-level sits $0.05$
above the chemical potential. The converged
hybridization $V_j=0.37$ mixes it with the conduction band, and a gap opens
between the top of the lower band, at $0.007$, and the bottom of the upper
band, at $0.19$. Close to the gap the bands are flat and mostly red: these
are the heavy quasiparticles.

??? example "Code of this figure (documentation/figures/pseudofermions_kondo.py)"

    ```python
    --8<-- "pseudofermions_kondo.py"
    ```

The keywords and caveats are:

- `J`: the Coqblin-Schrieffer coupling above.
- `filling`: sets the chemical potential once, from the bands with $V=0$, and
  then keeps it fixed. The number of electrons can change while $V$ and
  $\lambda$ converge. The constraint $\langle n_f\rangle=1$ is imposed by
  $\lambda_j$, not by `filling`.
- `mf`: the initial $(V,\lambda)$. $V=0$ is always a self-consistent solution,
  like the trivial solution of the BCS gap equation. A run without a seed
  starts there and stays there, even when a hybridized solution exists. So a
  nonzero seed, as above, is needed. When both exist, the hybridized state
  has the lower energy.
- `T`: the temperature of the Fermi-Dirac distribution, 0.02 by default.

Do not choose a filling that puts the chemical potential inside the flat
f-level of the problem with $V=0$. All the f-orbitals are then at the same
energy $\lambda$, and a wide range of fillings, roughly 0.25 to 0.75 for one
conduction orbital per site, give the same ill-defined starting point. The
`filling=0.15` above keeps the chemical potential in the conduction band.

At zero temperature the large-$N$ theory gives a hybridization for any $J$,
with a Kondo scale $T_K=D\,e^{-1/(J\rho)}$, where $D$ is the bandwidth and
$\rho$ the density of states. At the finite temperature `T` of the loop this
becomes a crossover: below a threshold in $J$, which depends on $T$, $V=0$ is
the only solution, and at the threshold $V$ jumps to a value of order one
instead of growing continuously from zero.

A Zeeman field acts exactly on both the conduction electrons and the local
moments, since both terms are bilinear. As for spinons, we add it with
`h.add_zeeman()` or `h.add_exchange()` on the `KondoLatticeHamiltonian`,
before the mean-field calculation:

```python
h = KondoLatticeHamiltonian(hc)
h.add_zeeman([0., 0., 0.05])
h2 = h.get_mean_field_hamiltonian(J=1.5, filling=0.15, nk=150, mf=seed)
```

The field acts on every site, conduction and f. To put it on only one of the
two, give a function of the position instead of a vector (the f-sites are
shifted in $z$). The constraint $\langle n_f\rangle=1$ stays exact in a field.
The field competes with the Kondo singlet: at fixed $J$, $|V|$ decreases as
the field grows, and a large enough field destroys the hybridized state. There
the loop does not converge and the call raises an error, instead of giving a
small nonzero $V$.
This is physics, not a numerical failure: it is the same decay to the $V=0$
solution that a small $J$ gives.

See `examples/kondolattice/chain/main.py` for a runnable script, and the
notebook
`jupyter-notebooks/functionalities/interacting_mean_field_hamiltonians/14_kondo_lattice.ipynb`.

## Tunneling spectrum of the Kondo lattice

We will now see the tunneling spectrum of the heavy-fermion state. A tip above
a site can tunnel into the conduction electron, with amplitude $t_c$. It can
also tunnel through a spin flip of the local moment, which in the mean field
is a tunneling into the f fermion, with amplitude $t_f$. The tip then couples
to the combination

$$
\psi_j=t_c\,c_j+t_f\,f_j ,
$$

and the conductance is its spectral function (arXiv:0910.1138 and
arXiv:1001.4530),

$$
\frac{dI}{dV}(eV)\propto t_c^2N_{cc}(eV)+t_f^2N_{ff}(eV)+2t_ct_fN_{cf}(eV) ,
$$

where $N_{cc}$ and $N_{ff}$ are the local density of states of the
conduction and of the f fermion, and $N_{cf}=-\mathrm{Im}\,G_{cf}/\pi$ is
their interference. The two paths interfere, and this is what gives the
asymmetric line shape that STM sees in heavy-fermion compounds.
`h2.get_cotunneling_didv()` computes it for a converged `h2`:

```python
import numpy as np
from pyqula import geometry
from pyqula.kondolattice import KondoLatticeHamiltonian
hc = geometry.chain().get_hamiltonian(has_spin=True)
h = KondoLatticeHamiltonian(hc)
h2 = h.get_mean_field_hamiltonian(J=1.5, filling=0.15, nk=150,
        mf=([0.3+0.0j], [0.0]))
energies = np.linspace(-0.4, 0.6, 201)
(es, didv, parts) = h2.get_cotunneling_didv(energies=energies, tc=1.0, tf=0.3,
        nk=2000, delta=0.01, T=0.01, site=0, return_parts=True)
```

The energy is measured from the chemical potential, and a positive energy is
tunneling into empty states. `didv` is the spectrum of the site, summed over
the two spins, and `parts` has its three terms, `"cc"`, `"ff"` and `"cf"`.
Here $V=0.374$, and the hybridization opens a gap between $eV=0$ and
$eV=0.2$. With $t_f=0$ the spectrum is the conduction density of states. It
has a peak of height $0.72$ at the upper edge of the gap and no peak at the
lower edge, where it is about $0.3$ and the states are mostly f. With
$t_f=0.3$ the upper peak grows to $2.1$, because there the two paths add.
With $t_f=-0.3$ they cancel at the upper edge, and the peak is at the lower
edge, with height $2.5$. So the sign of $t_f/t_c$ decides on which side of the gap the peak is.
Inside the gap only the broadening is left, $0.08$ at $eV=0.1$.

The keywords are:

- `tc`, `tf`: the two tunneling amplitudes.
- `J`: when it is given, `tf` is the amplitude of the spin-flip tunneling
  itself, and the amplitude into the f fermion of each site is
  $2t_fV_j^*/J$, so it follows the hybridization of the site. Give the $J$ of
  the mean-field calculation, the Coqblin-Schrieffer coupling above.
- `site`: the Kondo site under the tip, or a list. By default every site.
- `energies`: a uniform grid. By default `ne` points between `-emax` and
  `emax`.
- `delta`: the broadening of the levels.
- `T`: the temperature of the tip. It must be larger than the energy step.
- `nk`: the k-mesh. The level spacing must be smaller than `delta`.

This is the spectrum of the mean field, so above the temperature where $V$
vanishes it is only the conduction band. See
`examples/kondolattice/cotunneling_didv/main.py` for a runnable script, which
also compares the gap and the line shape with their closed forms.

## Composite fermion of the Kondo lattice

We will now see the spectral function of the Kondo lattice resolved in
momentum, for the conduction electron and for the composite fermion. This is
the comparison that arXiv:2302.09701 makes between matrix product states and
the mean field of the Kondo chain. That work writes the Kondo term with the
Pauli matrices,

$$
H=\sum_k\epsilon_k c^\dagger_kc_k
+J'\sum_j \left(c^\dagger_j\vec\sigma c_j\right)\cdot\vec S_j ,
$$

so its coupling is $J'=J/2$ in terms of the Coqblin-Schrieffer $J$ of this
page. There are two ways to add an electron to a site. We can add a
conduction electron, with $c^\dagger_j$. We can also add an electron and flip
the moment of the site at the same time. The operator of the second process
is what is called the composite fermion,

$$
F_{j\alpha}=\frac{2}{3}\sum_\beta
\left(\vec\sigma\cdot\vec S_j\right)_{\alpha\beta}c_{j\beta} .
$$

It has charge $e$ and spin one half, as the electron. It is also the operator
that the Kondo term couples to the electron, because the equation of motion
of the electron is
$i\partial_t c_k=\epsilon_kc_k+\tfrac{3}{2}J'F_k$. The two operators give a
$2\times2$ matrix of retarded Green functions and its spectral functions,

$$
G_{AB}(k,t)=-i\theta(t)\langle\{A_k(t),B^\dagger_k(0)\}\rangle ,
\qquad
A_{cc}=-\frac{1}{\pi}\mathrm{Im}\,G_{cc},
\qquad
A_{FF}=-\frac{1}{\pi}\mathrm{Im}\,G_{FF} ,
$$

with $A$ and $B$ equal to $c$ or $F$. $A_{cc}$ is what ARPES measures.
$A_{FF}$ is the weight of the states that we reach through a spin flip.

In the mean field the moment is made of f fermions, and the Kondo term is
the hybridization $V_j$. The same equation of motion is then
$i\partial_t c_k=\epsilon_kc_k+V^*f_k$, meaning that the composite fermion
is the f fermion times an amplitude,

$$
F_j=\frac{2V_j^*}{3J'}f_j=\frac{4V_j^*}{3J}f_j .
$$

So the mean field gives this matrix from the two hybridized bands, and
`h2.get_composite_spectral_function()` computes it along a k-path:

```python
import numpy as np
from pyqula import geometry
from pyqula.kondolattice import KondoLatticeHamiltonian
hc = geometry.chain().get_hamiltonian(has_spin=True)
h = KondoLatticeHamiltonian(hc)
Jp = 2.0 # J' of the Kondo term above, in units of t
h2 = h.get_mean_field_hamiltonian(J=2*Jp, filling=0.5, nk=200,
        mf=([Jp+0.0j], [0.0]))
energies = np.linspace(-6., 6., 300)
(ks, es, A) = h2.get_composite_spectral_function(energies=energies, nk=200,
        delta=0.05, hybridization_scale=1.5)
```

`ks` are the k-points and `es` the energies, measured from the chemical
potential. `A["cc"]` and `A["FF"]` are the two spectral functions, arrays
over momentum and energy, averaged over the two spins. `A["cF"]` is the mixed
one. `A["gF"]` is $g_F=1/[G^{-1}]_{FF}$, the Green function of the composite
fermion without its hybridization with the electrons.

Here the conduction band is half filled, which is the Kondo insulator. For a
band with particle-hole symmetry, as the one of the chain, half filling is
the one case where a chemical potential inside the flat f-level is safe: the
chemical potential and $\lambda$ are both zero by symmetry. The loop gives $V=1.88$, and `hybridization_scale=1.5` multiplies it by
$3/2$ before the spectrum is computed, for the reason given below.

![Spectral functions of the conduction electron and of the composite fermion of the half-filled Kondo chain, over momentum and energy.](img/pseudofermions_composite.png#only-light)
![Spectral functions of the conduction electron and of the composite fermion of the half-filled Kondo chain, over momentum and energy.](img/pseudofermions_composite-dark.png#only-dark)

The two spectral functions show the same two bands,
$E_\pm(k)=\tfrac12[\epsilon_k\pm\sqrt{\epsilon_k^2+4V^2}]$, with
$\epsilon_k=2t\cos k$ and the rescaled $V=2.82$. The upper band goes from
$3.99$ at $k=0$ to $1.99$ at $k=\pi$, and the lower band is its mirror image.
The gap is indirect, $3.98$ between $k=0$ and $k=\pi$. What separates the two
panels is the weight. At $k=0$ the upper band is mostly conduction electron,
with a peak of $4.1$ in $A_{cc}$ against $2.1$ in $A_{FF}$. At $k=\pi$ it is
the opposite, and at $k=\pi/2$, where $\epsilon_k=0$, the two are equal. In
the mean field $g_F$ is one pole at the f-level for every $k$, so the
composite fermion has no dispersion of its own.

??? example "Code of this figure (documentation/figures/pseudofermions_composite.py)"

    ```python
    --8<-- "pseudofermions_composite.py"
    ```

The mean field has the right two bands, but its hybridization is too small.
At strong coupling the loop gives $V=J'$, and the exact energy to add an
electron to a Kondo singlet is $3J'/2$. So arXiv:2302.09701 multiplies $V$ by
$3/2$ when it compares with matrix product states, and
`hybridization_scale=1.5` does the same here. With it, the upper band at
$J'=2t$ is at $1.99$, $2.82$ and $3.99$ for the three momenta of a ring of 4
sites. The exact poles of $A_{FF}$ of that ring are at $2.14$, $2.88$ and
$3.96$. This rescale is put by hand, it is not a result of the mean field.

The keywords are:

- `kpath`, `nk`: the k-points, as in `h.get_bands()`.
- `energies`: the energies. By default `ne` points between `-emax` and `emax`.
- `delta`: the broadening of the levels.
- `hybridization_scale`: a factor that multiplies $V$. 1 is the mean field
  itself and 1.5 is the comparison above. The f-level is not changed, so the
  f occupation stays at one only at half filling.
- `J`: by default the composite fermion is the f fermion, $F=f$, and $A_{FF}$
  integrates to one. With the $J$ of the mean-field calculation,
  $F=(4V^*/3J)f$ and $A_{FF}$ integrates to $|4V/3J|^2$.
- `site`: a Kondo site of the unit cell, or a list. By default the average
  over the sites of the cell.
- `green`: with `True`, `A` has the complex Green functions `"cc"`, `"cF"`,
  `"Fc"` and `"FF"` instead of the spectral functions.

With `J` and `green=True` the Green functions obey the relations that the
equation of motion gives for the exact problem,
$G_{cc}=g_c+g_c\tfrac32J'G_{FF}\tfrac32J'g_c$ and
$G_{cF}=g_c\tfrac32J'G_{FF}$, with $g_c=1/(\omega-\epsilon_k)$. See
`examples/kondolattice/composite_spectral_function/main.py` for a runnable
script, which checks them and compares with the exact energy of one site.

## Spin susceptibility of the local moments

We will now see the dynamical spin susceptibility of the local moments of
the Kondo lattice, which is what inelastic neutron scattering measures. We
take the transverse response between the moments $i$ and $j$ of the unit
cell,

$$
\chi_{ij}(q,\omega)=\langle\langle S^-_j;S^+_i\rangle\rangle_{q,\omega} ,
$$

where $S^+$ creates the excitation, on the moment of the first index, and
$S^-$ removes it, on the moment of the second. In the mean field the moment is
$\vec S_j=\tfrac12f^\dagger_j\vec\sigma f_j$, so a spin flip creates a
particle and a hole in the hybridized bands. The susceptibility is then the
product of two f propagators. For the half-filled chain it is

$$
\chi(q,\omega)=\frac{1}{N}\sum_k w^-_k\,w^+_{k+q}
\left[\frac{1}{\omega-\Delta_{k,q}}-\frac{1}{\omega+\Delta_{k,q}}\right],
\qquad
\Delta_{k,q}=E_+(k+q)-E_-(k) ,
$$

where $w^\pm_k$ is the weight of the f fermion in the band $E_\pm(k)$. This
is a continuum of pairs of quasiparticles with spin one half, without a sharp
mode. You can think of it as the spin flip breaking in two halves.

The mean field leaves out the interaction between the particle and the hole.
arXiv:2302.09701 puts it back as a residual interaction $U'(q)$ in the random
phase approximation,

$$
\chi_{\rm RPA}(q,\omega)=\frac{1}{1/\chi(q,\omega)-U'(q)} .
$$

A negative $U'$ is an attraction, and it binds the particle and the hole in
a magnon below the continuum. `h2.get_moment_spinchi()` computes the two
susceptibilities, for the same half-filled chain as above:

```python
import numpy as np
from pyqula import geometry
from pyqula.kondolattice import KondoLatticeHamiltonian
hc = geometry.chain().get_hamiltonian(has_spin=True)
h = KondoLatticeHamiltonian(hc)
Jp = 2.0 # J' of the Kondo term, in units of t
h2 = h.get_mean_field_hamiltonian(J=2*Jp, filling=0.5, nk=200,
        mf=([Jp+0.0j], [0.0]))
energies = np.linspace(0., 6., 200)
(es, chi0) = h2.get_moment_spinchi(q=[0.5,0.,0.], energies=energies,
        nk=200, delta=0.05)
(es, chi) = h2.get_moment_spinchi(q=[0.5,0.,0.], energies=energies,
        nk=200, delta=0.05, U=-2.0)
```

`q` is in units of the reciprocal lattice vector, so this is $q=\pi$. `chi0`
is the mean-field response and `chi` the one with $U'=-2t$. Each one is a
matrix between the moments of the unit cell for every frequency, $1\times1$
here. The spectral weight at positive frequency is the imaginary part,
`chi[:,0,0].imag`, and the real part at zero frequency is negative.

![Imaginary part of the spin susceptibility of the moments of the Kondo chain over momentum and frequency, in the mean field and with the residual interaction.](img/pseudofermions_momentchi.png#only-light)
![Imaginary part of the spin susceptibility of the moments of the Kondo chain over momentum and frequency, in the mean field and with the residual interaction.](img/pseudofermions_momentchi-dark.png#only-dark)

The figure repeats the two calls for 61 momenta, with $V=1.88$. In the mean
field there is only the continuum. It starts at the direct gap $2V=3.75$ at
$q=0$ and at the indirect gap $2.25$ at $q=\pi$. With $U'=-2t$ a sharp mode
appears below it, at $3.46$ for $q=0$ and at $1.96$ for $q=\pi$. This is the
magnon that the matrix product states of arXiv:2302.09701 show at $J'=2t$.

??? example "Code of this figure (documentation/figures/pseudofermions_momentchi.py)"

    ```python
    --8<-- "pseudofermions_momentchi.py"
    ```

$U'$ is a parameter of the comparison, the mean field does not give it. That
work takes $U'=-2t$ for $J'=2t$, and $U'(q)=-0.75t\sin(q/2)$ for $J'=0.9t$,
where the magnon merges with the continuum at small $q$. The second one is
`U=lambda q: -0.75*np.sin(np.pi*q[0])` here. The approximation is valid while
$1-U'\chi(q,0)>0$. When this reaches zero the magnon is at zero frequency,
meaning that the mean field is unstable to magnetic order with wavevector
$q$. For $J'=0.9t$ and $U'=-0.75t$ at $q=\pi$ this number is $-0.01$ with the
$V$ of the loop, and $0.54$ with `hybridization_scale=1.5`.

The mean field also loses spectral weight. The local response, integrated
over positive frequency, gives $1/4$ for each moment, and a spin one half has
$\langle S^-S^+\rangle=1/2$. The reason for this is that the constraint of
one fermion per site holds only on average.

The keywords are:

- `q`: the momentum, in units of the reciprocal lattice vectors. `None` is
  the local response, the average over the momenta of the k-mesh. Its cost
  grows as the square of the number of k-points, so take a small `nk` in two
  and three dimensions.
- `U`: the residual interaction $U'$. A number, a matrix between the moments
  of the unit cell, or a function of `q` that returns one of them. By default
  there is none.
- `nk`: the k-mesh.
- `delta`: the broadening of the frequency.
- `T`: the temperature of the occupations, equal to `delta` by default.
- `hybridization_scale`: the factor that multiplies $V$, as above.

A Zeeman field, a supercell and a hopping between the moments are all
included, because the response is built from the bands of the full
Hamiltonian. See `examples/kondolattice/moment_spinchi/main.py` for a
runnable script.

### Chern number of the magnon of the moments

We will now see that the magnon of the moments can carry a Chern number,
with `h2.get_moment_magnon_chern()`. A mode of the RPA response is a zero of
the kernel

$$
K(\mathbf q,\omega)=\chi(\mathbf q,\omega)^{-1}-U'(\mathbf q) ,
$$

a matrix between the moments of the unit cell. At the energy of the mode
$K$ has a null vector $\psi_i(\mathbf q)$, the amplitude of the magnon on the
moment $i$. This vector plays the role of the Bloch state, and the Chern
number of the magnon band is the sum of its Berry fluxes through the
plaquettes of a mesh of $\mathbf q$, over $2\pi$, with the sign of
`h.get_chern()`. The zeros are counted in a window of frequencies
$(\omega_a,\omega_b)$ that you choose: both ends in gaps of the magnon
spectrum at every $\mathbf q$, $\omega_a$ above zero and $\omega_b$ below the
continuum.

With one moment per cell the kernel is a number, so its null vector carries
no Chern number, and in the raising channel, the default, the result is zero
by construction. The chain above has no Chern number for a second reason: it
is one dimensional, and the method raises `ValueError`. So we take the
honeycomb lattice, two moments per cell, with $J=4$ at half filling, a
Haldane term $t_2=0.2$ on the conduction electrons, which breaks time
reversal, and $U'=-4$:

```python
from pyqula import geometry
from pyqula.kondolattice import KondoLatticeHamiltonian
hc = geometry.honeycomb_lattice().get_hamiltonian(has_spin=True)
hc.add_haldane(0.2) # breaks time reversal of the conduction electrons
h = KondoLatticeHamiltonian(hc) # two moments per unit cell
h2 = h.get_mean_field_hamiltonian(J=4.0, filling=0.5, nk=12,
        mf=([1.0+0j]*2, [0.0]*2))
c = h2.get_moment_magnon_chern((0.3, 1.43), U=-4.0, nk=8, nq=12)
```

`c` is $+1$. One magnon is bound below the continuum: on this mesh its band
goes from $0.590$ to $1.357$, the next band starts at $1.506$ and the
continuum at $1.695$, so the window $(0.3,1.43)$ holds this band alone. With
`full_output=True` the record of `h.get_magnon_chern()` comes back: `gap` is
$0.073$, here the distance to the upper end of the window, and `bound` is
$0.75$, the distance below the continuum. The integer is the same with
`nq=6` and with `nk=10`, and $t_2=-0.2$ gives $-1$. For $t_2=\pm0.1$ the band
reaches $1.465$ and the next one starts at $1.547$, so the window is
$(0.3,1.506)$, and the result is $\pm1$. So the Chern number is the sign of
$t_2$ in every case we ran; this is measured, not derived. With $t_2=0$ the
two magnon bands touch at K and K', at $1.5305$, and there is no Chern
number.

![Energy and Berry curvature of the magnon of the moments of the honeycomb Kondo lattice with a Haldane term, over the Brillouin zone, on a mesh of 12 by 12 points.](img/pseudofermions_kondochern.png#only-light)
![Energy and Berry curvature of the magnon of the moments of the honeycomb Kondo lattice with a Haldane term, over the Brillouin zone, on a mesh of 12 by 12 points.](img/pseudofermions_kondochern-dark.png#only-dark)

The figure reads the energies and the fluxes of the record of the snippet.
The magnon has its minimum, $0.590$, at $\Gamma$, the corners of the map,
and its maximum, $1.357$, at K and K', at $(1/3,1/3)$ and $(2/3,2/3)$. Its
Berry curvature is positive and sits around K and K', the points where the
two bands touch when $t_2=0$. The largest flux of a plaquette is $0.61$,
about $0.19\pi$.

??? example "Code of this figure (documentation/figures/pseudofermions_kondochern.py)"

    ```python
    --8<-- "pseudofermions_kondochern.py"
    ```

$U'$ is a parameter of the model, as in the section above, so the integer is
a property of the model with that $U'$. With $U'=-1.5$ or $-2.5$ the magnon
reaches the continuum and there is no window. The moments have spin-rotation
symmetry here, so the magnon is a spin triplet, the same mode for $S^+$,
$S^-$ and $S_z$. The raising channel holds one copy of it. The full channel,
`channel="full"`, holds three copies with the same vector, so the flux of
each plaquette is three times larger, and the integer it returns depends on
the mesh: for $t_2=0.2$ it gives $3$, $1$, $1$ and $3$ at `nq=6`, 8, 10 and
12, with no error, while the raising channel gives $1$ on each of these
meshes. So a spin triplet is read in the raising channel.

The keywords are:

- `window`: the frequencies $(\omega_a,\omega_b)$ between which the zeros
  are counted.
- `U`: the residual interaction $U'$, as in `h2.get_moment_spinchi()`.
- `bands`: the zeros of the window, counted from the lowest, a contiguous
  list; all of them by default.
- `nq`, `shift`: the mesh of $\mathbf q$, 12 points along each direction by
  default.
- `nk`: the k-mesh of the susceptibility.
- `T`: the temperature of the occupations, zero by default.
- `channel`: `"raising"` by default, `"lowering"` or `"full"`.
- `hybridization_scale`: the factor that multiplies $V$, as above.
- `full_output`: return the record instead of the number.

See `examples/kondolattice/moment_magnon_chern/main.py` for a runnable
script.

## Kondo lattices in two dimensions and in finite systems

We will now see that the two calculations above are not specific to the
chain. `h2.get_composite_spectral_function()` and `h2.get_moment_spinchi()`
take a Kondo lattice of any dimensionality, from a finite system to a 3D
crystal, with any number of Kondo sites in the unit cell. The momenta
`kpath` and `q` are in units of the reciprocal lattice vectors, so only
their number of components changes. Let us take the square lattice at half
filling, with the same $J'=2t$:

```python
import numpy as np
from pyqula import geometry
from pyqula.kondolattice import KondoLatticeHamiltonian
hc = geometry.square_lattice().get_hamiltonian(has_spin=True)
h = KondoLatticeHamiltonian(hc)
Jp = 2.0 # J' of the Kondo term, in units of t
h2 = h.get_mean_field_hamiltonian(J=2*Jp, filling=0.5, nk=40,
        mf=([Jp+0.0j], [0.0]))
kpath = [[0.,0.,0.], [0.5,0.,0.], [0.5,0.5,0.]] # Gamma, X and M
(ks, es, A) = h2.get_composite_spectral_function(kpath=kpath,
        energies=np.linspace(-6., 6., 300), delta=0.05)
(ws, chi) = h2.get_moment_spinchi(q=[0.5,0.5,0.],
        energies=np.linspace(0., 5., 150), nk=30, delta=0.05, U=-2.0)
```

`A["cc"]` and `A["FF"]` have one row for each of the three momenta, and
`chi` is the susceptibility at M with the residual interaction $U'=-2t$.

![Spectral function of the composite fermion and spin susceptibility of the moments of the Kondo square lattice along the path Gamma, X, M, Gamma.](img/pseudofermions_kondo2d.png#only-light)
![Spectral function of the composite fermion and spin susceptibility of the moments of the Kondo square lattice along the path Gamma, X, M, Gamma.](img/pseudofermions_kondo2d-dark.png#only-dark)

The figure repeats the two calls along the path $\Gamma$, X, M, $\Gamma$.
The loop gives $V=1.79$, and the bands are again
$E_\pm(k)=\tfrac12[\epsilon_k\pm\sqrt{\epsilon_k^2+4V^2}]$, now with
$\epsilon_k=2t(\cos k_x+\cos k_y)$. The upper band is at $4.68$ at $\Gamma$,
at $1.79$ at X and at $0.68$ at M. The composite fermion has little weight
at $\Gamma$, $0.15$ of the one of the conduction electron, the same weight
at X, where $\epsilon_k=0$, and most of the weight at M. The gap is indirect,
$1.36$ between M and $\Gamma$. In the susceptibility the magnon is at $3.32$
at $\Gamma$, at $1.98$ at X and at $1.21$ at M, below the continuum, which
starts at $1.36$ at M. Its minimum is at M, the wavevector of the
antiferromagnetic order of the square lattice.

??? example "Code of this figure (documentation/figures/pseudofermions_kondo2d.py)"

    ```python
    --8<-- "pseudofermions_kondo2d.py"
    ```

A finite system has no momentum. The spectral functions are then the local
ones of each Kondo site, which is what STM measures, and the susceptibility
is a matrix between the moments. Let us take a honeycomb flake of 24 sites:

```python
import numpy as np
from pyqula import islands
from pyqula.kondolattice import KondoLatticeHamiltonian
g = islands.get_geometry(name="honeycomb", n=2, nedges=6) # 24 sites
h = KondoLatticeHamiltonian(g.get_hamiltonian(has_spin=True))
n = len(g.r)
Jp = 2.0 # J' of the Kondo term, in units of t
h2 = h.get_mean_field_hamiltonian(J=2*Jp, filling=0.5,
        mf=([Jp+0.0j]*n, [0.0]*n))
(ks, es, A) = h2.get_composite_spectral_function(
        energies=np.linspace(-6., 6., 301), delta=0.05, site=list(range(n)))
(ws, chi) = h2.get_moment_spinchi(energies=np.linspace(0., 6., 301),
        delta=0.05)
```

`A["FF"]` has the shape (24, 1, 301): one row for each site, one k-point and
the energies. `chi` has the shape (301, 24, 24). Here $V$ goes from $1.82$
to $1.88$ between the sites, because the sites of the edge have fewer
neighbors, and the gap of the flake is $1.86$. In a finite system `kpath`
and `q` are ignored.

With several Kondo sites the residual interaction can be a matrix between
the moments of the cell, and the dressing is
$\chi_{\rm RPA}=\chi(1-U'\chi)^{-1}$. A band without particle-hole symmetry,
as the one of the triangular lattice, needs a filling away from one half,
as explained in the section on the Kondo lattice. See
`examples/kondolattice/composite_fermion_2d/main.py` and
`examples/kondolattice/composite_fermion_0d/main.py` for runnable scripts.
