# Files written to the working directory

Most of the routines that compute an observable also write it to a file in the working
directory, `BANDS.OUT` for the band structure, `DOS.OUT` for the density of states,
`CHERN.OUT` for the Chern number and so on, which is convenient when a calculation runs as a
script and is plotted afterwards, and less so inside a notebook or a larger program, where
the returned arrays are all that is needed and the files only pile up. Each of these routines
takes a `write` keyword with a default of its own, and we will now see how to set it for all
of them at once, with `filewrite.set_write()`

```python
from pyqula import geometry
from pyqula import filewrite
filewrite.set_write(False) # no routine writes its files from now on
g = geometry.honeycomb_lattice() # honeycomb lattice
h = g.get_hamiltonian() # first-neighbor tight-binding Hamiltonian
(k,e) = h.get_bands() # the band structure, and no BANDS.OUT
(k,e) = h.get_bands(write=True) # a write given in the call wins over the switch
filewrite.set_write(None) # back to the default of each routine
```

The order is always the same: a `write` given in the call, then the switch, and only when
neither is set, the default of the routine. So `filewrite.set_write(True)` makes every
routine write, including the few that do not by default, such as `topology.chern_density()`,
and `write=False` in a single call keeps that call quiet whatever the switch says. The
returned arrays do not depend on any of this, only the files do. The switch reaches every
routine that takes a `write` keyword; the methods whose only purpose is to write a file, such
as `g.write()` or `h.write_hopping()`, write whenever they are called.
