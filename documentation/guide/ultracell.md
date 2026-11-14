# Long-range modulations

In this page we see how to compute a crystal whose potential, exchange field
or pairing changes slowly over many unit cells, with the ultracell. We first
compute the spectral function of graphene under a slow potential. Then we
compute the local density of states and the charge over a modulation. At the
end we keep only the low-energy states of a unit cell with many orbitals.

## A modulation over many cells

A charge density wave, a superlattice defined by a gate or a magnetic texture
can repeat only every tens or hundreds of unit cells. The atoms are still
periodic, and only a slow envelope changes from cell to cell. The smallest
cell that holds the whole modulation is a supercell of $N=n_1n_2n_3$ unit
cells. This is what is called the ultracell.

Without the modulation, the eigenstates of the ultracell at a momentum
$\mathbf k_S$ are Bloch states of the unit cell. They have the $N$ momenta

$$
\mathbf p_{\mathbf q} = \frac{\mathbf k_S+\mathbf q}{\mathbf n},
\qquad \mathbf q\in\mathbb Z_{n_1}\times\mathbb Z_{n_2}\times\mathbb Z_{n_3},
$$

component by component, in reduced coordinates. We write the modulated
ultracell, $H_S=H_0+\delta H$, in these states:

$$
H_{\mathbf q m,\mathbf q'm'} = \delta_{\mathbf q\mathbf q'}\,\delta_{mm'}\,
\varepsilon_m(\mathbf p_{\mathbf q})
+ \langle \mathbf p_{\mathbf q},m|\,\delta H\,|\mathbf p_{\mathbf q'},m'\rangle .
$$

Here $\varepsilon_m(\mathbf p)$ are the bands of the unit cell,
$|\mathbf p,m\rangle$ its Bloch states, and $\delta H$ the modulation. The
diagonal is the band structure of the unit cell. The modulation couples two
momenta that differ by one of its wavevectors $\mathbf Q$. With every state
kept, this matrix is exactly the supercell, written in another basis. This is
the ultra long-range method of
[arXiv:2008.12573](https://arxiv.org/abs/2008.12573), written for a
tight-binding basis.

We build the ultracell as a supercell, and add the modulation to it with the
usual methods. Then `h.get_ultracell(hs)` solves it, with `h` the unit cell.
For concreteness we take graphene and a potential
$V(\mathbf r)=V_0\cos(2\pi x_1/n)$, with $x_1$ the coordinate along
$\mathbf a_1$ in units of the lattice vector. It has one period over $n=20$
cells and $V_0=0.3t$:

```python
import numpy as np
from pyqula import geometry
g = geometry.honeycomb_lattice() # unit cell of graphene
h = g.get_hamiltonian() # first-neighbor hopping, t=1
n = 20 # period of the modulation, in unit cells along a1
hs = h.get_supercell([n,1]) # the ultracell, written as a supercell
b1 = np.linalg.inv(np.array([g.a1,g.a2,g.a3]).T)[0] # b1.r is the coordinate along a1
V0 = 0.3 # amplitude of the potential, in units of t
hs.add_onsite(lambda r: V0*np.cos(2*np.pi*b1.dot(r)/n)) # one period over n cells
u = h.get_ultracell(hs) # hs solved in the Bloch states of graphene
(k,e,a) = u.get_kdos_bands(energies=np.linspace(-1.5,1.5,301),nk=400,delta=0.02)
```

`b1.dot(r)` is the coordinate $x_1$ of the position `r`. `u.get_kdos_bands()`
returns the unfolded spectral function $A(\mathbf k,\omega)$ along the k-path
of the unit cell, in the same form as `h.get_kdos_bands()`: `k` is the
position along the path, `e` the energy and `a` the spectral function. Each
basis state has one momentum of the unit cell, so the unfolded weight comes
directly from the coefficients of the states. It is the weight per unit cell:
without the potential, `a` is exactly `h.get_kdos_bands()` of graphene. This
is what ARPES measures on a crystal with a superlattice.

![Unfolded spectral function of graphene under a potential with one period over 20 unit cells.](img/ultracell_unfolding.png#only-light)
![Unfolded spectral function of graphene under a potential with one period over 20 unit cells.](img/ultracell_unfolding-dark.png#only-dark)

The potential mixes each Bloch state with the states at
$\mathbf k\pm\mathbf Q$, with $\mathbf Q=\mathbf b_1/n$. Between $\Gamma$ and
K each band keeps most of its weight, and gets weaker replicas on both sides,
about a third as bright. Near M the band of graphene, at $\omega=\pm t$,
spreads into a ladder of discrete levels from $0.72t$ to $1.28t$, just
inside $t\pm V_0$. At M the band is a saddle point and changes slowly with
momentum, so the potential, and not the kinetic energy, sets these levels.
Near K and K' the Dirac cone and its replicas cross.

??? example "Code of this figure (documentation/figures/ultracell_unfolding.py)"

    ```python
    --8<-- "ultracell_unfolding.py"
    ```

The keywords that change the result:

- `emax=None` in `h.get_ultracell()`: keep the states of the unit cell with
  energy up to `emax`; `None` keeps all of them, which is the supercell
  exactly (see [Keeping only the low-energy states](#keeping-only-the-low-energy-states))
- `energies`, `delta`, `nk` and `kpath` in `u.get_kdos_bands()`: the same as
  in `h.get_kdos_bands()`, with the k-path of the unit cell

The modulation can be anything that pyqula adds to a supercell: an onsite
potential, an exchange field with `hs.add_exchange()`, a pairing with
`hs.add_swave()`, or hoppings that change from cell to cell. The supercell
must be diagonal, $n_1\times n_2\times n_3$ copies of the unit cell along its
own lattice vectors, and it must keep every site.

See `examples/2d/ultracell/main.py`.

## Local density of states and charge

We will now see what STM sees over a modulation in two dimensions, and the
charge that the modulation induces. We take spinless graphene and a potential
with threefold symmetry and a period of $n=30$ cells along $\mathbf a_1$ and
$\mathbf a_2$,

$$
V(\mathbf r) = V_0\left[\cos\frac{2\pi x_1}{n}+\cos\frac{2\pi x_2}{n}
+\cos\frac{2\pi (x_1-x_2)}{n}\right],
$$

which goes from $-1.5V_0$ to $3V_0$, with $V_0=0.1t$:

```python
import numpy as np
from pyqula import geometry
g = geometry.honeycomb_lattice() # unit cell of graphene
h = g.get_hamiltonian(has_spin=False) # spinless, first-neighbor hopping t=1
n = 30 # period of the modulation, in unit cells along a1 and a2
hs = h.get_supercell([n,n]) # the ultracell, written as a supercell
B = np.linalg.inv(np.array([g.a1,g.a2,g.a3]).T) # B@r are the coordinates along a1, a2
V0 = 0.1 # amplitude of each plane wave, in units of t
def V(r): # a potential with the period of the ultracell and threefold symmetry
    x1,x2 = 2*np.pi*(B@r)[0:2]/n
    return V0*(np.cos(x1) + np.cos(x2) + np.cos(x1-x2))
hs.add_onsite(V) # add it to the ultracell
u = h.get_ultracell(hs) # hs solved in the Bloch states of graphene
(x,y,d) = u.get_ldos(e=0.0,delta=0.03,nk=2) # LDOS at the Dirac energy
dn = u.get_vev(nk=2) - 0.5 # extra charge on each site, Fermi level at zero
(Q,f) = u.get_harmonics(dn) # its Fourier components, per unit-cell site
```

`u.get_ldos()` returns the positions `x`, `y` of the sites of the ultracell
and the local density of states `d` at the energy `e`. It has the
normalization of `h.get_ldos()` of the supercell. `u.get_vev()` returns the
occupation of each site with the Fermi level at zero energy, so `dn` is the
charge each site gains or loses with respect to half filling. With an
operator, for example `u.get_vev(operator="sz")`, it returns the expectation
value of that operator on each site. `u.get_harmonics()` gives the Fourier
components $n_{\mathbf Q}$ of a quantity over the cells, with one column per
site of the unit cell, and `Q` the wavevectors in reduced coordinates of the
unit cell.

![The potential and the local density of states at zero energy over four ultracells of graphene.](img/ultracell_ldos.png#only-light)
![The potential and the local density of states at zero energy over four ultracells of graphene.](img/ultracell_ldos-dark.png#only-dark)

The figure shows four ultracells. Locally, the electrons see graphene with
its Dirac point shifted to $V(\mathbf r)$, and the density of states of
graphene grows as $|\omega-V|$. The map follows this picture. The LDOS at
$\omega=0$ is 0.048 on the maxima of the potential, where $V=0.3t$. This is
about twice the 0.022 on its minima, where $V=-0.15t$. It is lowest, 0.020,
on the lines where $V$ vanishes, where only the broadening `delta` gives
weight. The extra charge follows $-V$: electrons gather where the potential
is low, and the correlation between `dn` and $-V$ is 0.997. Its six largest
harmonics are at the $\pm\mathbf Q$ of the three plane waves, all equal to
$9.6\times10^{-4}$, as the threefold symmetry requires. The ratio to the
amplitude $V_0/2$ of each wave, $0.019/t$, is at linear order the static
charge response of graphene at $\mathbf Q$ (see
[Charge response and RKKY](charge-response.md)).

??? example "Code of this figure (documentation/figures/ultracell_ldos.py)"

    ```python
    --8<-- "ultracell_ldos.py"
    ```

The keywords that change the result:

- `e` and `delta` in `u.get_ldos()`: the energy and the half width of the
  Lorentzian of each level
- `nk` in `u.get_ldos()`, `u.get_vev()` and `u.get_dos()`: the k-mesh of the
  ultracell in each direction. The momenta of the unit cell that it samples
  form a mesh $n$ times denser, so a large ultracell needs a small `nk`
- `operator` in `u.get_vev()`: the operator resolved on each site, a name
  such as `"sz"` or a matrix of the supercell

`u.get_dos()` gives the density of states per unit cell, with the same
keywords as `h.get_dos()`.

## Keeping only the low-energy states

For a unit cell with few orbitals, the ultracell has the size of the
supercell. Then a supercell with the
[kernel polynomial method](kpm.md) is just as fast. The ultracell pays off
when the unit cell has many orbitals and the physics lives in a few bands, as
in a moire cell or a Wannier Hamiltonian of a real material. Then we keep only
the states of the unit cell with energy up to `emax`. The matrix becomes much
smaller, and the one approximation is that the states above `emax` do not
take part in the modulated states.

For concreteness we take a $4\times4$ cell of the triangular lattice with a
moire potential, 16 orbitals, and a slow modulation of at most $0.1t$ over
$9\times9$ of these cells:

```python
import numpy as np
from pyqula import geometry, potentials
g = geometry.triangular_lattice().get_supercell(4) # a 4x4 cell of the triangular lattice
h = g.get_hamiltonian(has_spin=False) # 16 orbitals per cell
h.add_onsite(potentials.commensurate_potential(g,n=3,minmax=[0,2])) # moire potential, periodic on the cell
n = 9 # period of the slow modulation, in moire cells
hs = h.get_supercell([n,n]) # the ultracell, 81 moire cells
B = np.linalg.inv(np.array([g.a1,g.a2,g.a3]).T) # B@r are the coordinates along a1, a2
def V(r): # slow modulation, 0.1 t at most
    x1,x2 = 2*np.pi*(B@r)[0:2]/n
    return 0.05*(np.cos(x1) + np.cos(x2))
hs.add_onsite(V)
kS = [0.13,0.29,0.] # a Bloch momentum of the ultracell
e0 = np.linalg.eigvalsh(h.get_ultracell(hs).get_hk_gen()(kS)) # every state, exact
low = e0[e0<-1.9] # the levels we care about
for emax in [-1.8,-1.7,-1.5]:
    u = h.get_ultracell(hs,emax=emax) # keep the states up to emax
    m = u.get_hk_gen()(kS) # the ultracell matrix
    e = np.linalg.eigvalsh(m)
    print(emax,m.shape[0],np.max(np.abs(e[0:len(low)]-low)))
```

`u.get_hk_gen()` returns the function that gives the ultracell matrix at a
momentum of the ultracell. With every state the matrix has 1296 rows, the
size of the supercell. With `emax=-1.7` it has 150 rows, and the 89 levels
below $-1.9t$ agree with the full basis to $3\times10^{-4}t$. With
`emax=-1.8` they are off by $5\times10^{-3}t$, and with `emax=-1.5` by
$2\times10^{-4}t$. A state at a distance $\Delta E$ in energy enters the
modulated states with a weight of order $(\delta H/\Delta E)^2$. So we keep a
margin above the energies we look at, about twice the size of the modulation.
The cut is in energy and not in band index. Here the lowest band of the moire
cell reaches $-1.72t$ and the second one starts at $-1.95t$, so the levels
below $-1.9t$ need both bands. A degenerate level at the cut is kept whole.
You check the cut by raising `emax` until the numbers you need stop changing.
