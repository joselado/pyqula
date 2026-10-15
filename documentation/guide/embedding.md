# Single defects in infinite systems

Let us now see how a single point defect or impurity in an otherwise infinite, periodic system
is treated, with `embedding.Embedding`. A plain supercell calculation cannot do it without
periodizing the defect, so that the impurity sees its own replicas; the Green's function
embedding technique instead keeps the infinite pristine crystal and modifies a single cell. It
takes the pristine periodic Hamiltonian `h` together with a modified cell `m` describing the
defect, and gives the observables of the infinite system as perturbed by that single defect.
The defect itself is written with the same site-dependent onsite energy of the first chapter

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

`get_ldos` returns the real-space positions and the LDOS profile in a window of `nsuper` unit
cells around the defect, and what you see in it is how the infinite system responds to one
impurity: Friedel oscillations decaying away from it in a metal, or a bound state and in-gap
states induced by the impurity in a gapped system, the profile that STM measures around a
single adatom or vacancy (it also writes `LDOS.OUT`; pass `write=False` to suppress that).
`eb.get_dos()` gives the total DOS, `eb.multildos()` scans the LDOS over many energies
(written to a `MULTILDOS/` folder), `eb.get_gf()` returns the embedded Green's function itself
(with `operator=` it returns $AG$, the same convention `get_ldos` uses), and `eb.get_didv()`
computes transport through the embedded defect. See
`examples/embedding/single_impurity_1D/main.py` and
`examples/embedding/honeycomb_vacancy/main.py` for runnable versions, the other scripts under
`examples/embedding/` for further defect scenarios (vacancies, boundaries, Yu-Shiba-Rusinov
states...), and
`jupyter-notebooks/functionalities/spectral_functions/04_single_impurities_infinite.ipynb` for
the executed notebook. The momentum-space picture of the same defect, the quasiparticle
interference, is addressed in the Observables chapter.
