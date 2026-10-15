# Single defects in infinite systems

We now see how to compute the effect of a single defect, an impurity or a
vacancy, in an infinite crystal, with `embedding.Embedding`. A supercell
calculation cannot do this exactly: the defect is repeated in every
supercell, and it interacts with its own copies. The Green's function
embedding keeps the infinite pristine crystal and changes only one cell.

Let $H_0$ be the pristine periodic Hamiltonian, and $G_0(E)$ its local Green's
function on one unit cell,

$$G_0(E) = \frac{1}{N_k}\sum_k \left[E + i\delta - H_0(k)\right]^{-1}.$$

The rest of the infinite crystal acts on this cell as a self-energy
$\Sigma(E)$, defined by

$$G_0(E) = \left[E + i\delta - H_0^{\rm cell} - \Sigma(E)\right]^{-1},$$

where $H_0^{\rm cell}$ is the pristine intracell Hamiltonian. The defect
changes the Hamiltonian of this cell to $V$, and the rest of the crystal is
not changed, so the Green's function of the defective cell is the Dyson
equation

$$G(E) = \left[E + i\delta - V - \Sigma(E)\right]^{-1}.$$

This is exact for a defect inside one cell, in a crystal that is infinite in
all directions. The local density of states is
$\rho_n(E) = -\frac{1}{\pi}\,{\rm Im}\, G_{nn}(E)$, and around the defect it
is what STM measures near a single adatom or vacancy.

In pyqula, `h` is the pristine Hamiltonian and `m` is a Hamiltonian with the
defect; only its intracell matrix is used as $V$. Here the defect is an
onsite energy $1.0$ on one site of a chain:

```python
from pyqula import geometry
from pyqula import embedding
g = geometry.chain() # create the geometry
h = g.get_hamiltonian() # pristine, infinite Hamiltonian
hv = h.copy()
hv.add_onsite(lambda r: 1.0 if r[0]<0.01 else 0.0) # a single-site onsite defect
eb = embedding.Embedding(h,m=hv) # embed the defect in the infinite system
(x,y,d) = eb.get_ldos(energy=0.0,delta=1e-2,nsuper=200,nk=400) # LDOS around the defect
```

`eb.get_ldos()` returns the positions `x`, `y` of the sites in a window of
`nsuper` unit cells around the defect, and the LDOS `d` on each of them, at
the energy `energy`. In a metal, as this chain, you see Friedel oscillations:
the LDOS oscillates with wavevector $2k_F$ and decays away from the defect. In
a gapped system, a strong enough defect gives a bound state inside the gap,
and its LDOS is localized around the defect. The call also writes
`LDOS.OUT`; pass `write=False` to avoid it.

The other methods of the embedding object:

- `eb.get_dos()`: the total DOS of the window around the defect.
- `eb.multildos()`: the LDOS at many energies, written to the folder
  `MULTILDOS/`.
- `eb.get_gf()`: the Green's function $G(E)$ of the defective cell.
- `eb.get_didv()`: the differential conductance of an STM tip on the
  defect.

The momentum-space picture of the same defect is
[Quasiparticle interference](qpi.md). See
`examples/embedding/single_impurity_1D/main.py` and
`examples/embedding/honeycomb_vacancy/main.py` for runnable scripts, the other
scripts in `examples/embedding/` for vacancies, boundaries and
Yu-Shiba-Rusinov states, and
`jupyter-notebooks/functionalities/spectral_functions/04_single_impurities_infinite.ipynb`
for an executed notebook.
