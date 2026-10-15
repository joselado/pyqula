# Files written to the working directory

Many methods that compute a quantity also write it to a file in the working
directory: `BANDS.OUT` for the band structure, `DOS.OUT` for the density of
states, `CHERN.OUT` for the Chern number, and so on. This is useful for a
script that is plotted later, but in a notebook or inside a larger program
the returned arrays are enough, and the files are not needed. Each of these
methods has a keyword `write` with its own default, and
`filewrite.set_write()` sets it for all of them at once:

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

The rule is always the same: first the `write` given in the call, then the
switch, and only if neither is set, the default of the method. So
`filewrite.set_write(True)` makes every method write its file, also the few
that do not write by default, such as `topology.chern_density()`, and
`write=False` in one call keeps that call silent, whatever the switch says.
The returned arrays do not depend on this, only the files do. The methods
whose only job is to write a file, such as `g.write()` or
`h.write_hopping()`, always write.
