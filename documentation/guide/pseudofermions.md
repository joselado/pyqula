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
fermions.

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
state, so look at the converged hoppings.

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
  `"bond"`. The bond one takes an isotropic exchange and no Zeeman field.
- `ansatz`: the hopping pattern to hold, a Hamiltonian or a dictionary of
  hopping matrices. Without it the pattern can change during the loop.
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

The random phase approximation is uncontrolled, like the mean field. It keeps
the constraint of one fermion per site only on average, so the free spinons
have half of the total weight of a spin one half. On a lattice with nesting,
like the antiferromagnetic chain at $q=\pi$, the kernel is negative at low
temperature and the approximation cannot be used. `h2.get_magnon_bands()` gives
the energy of the sharp mode along a q-path. See
`examples/spinon/dirac_spin_liquid_rpa/main.py` for a runnable script.

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
V_j=-\frac{J}{2}\sum_a\langle f^\dagger_{ja}c_{ja}\rangle ,
$$

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
