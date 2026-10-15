# Parallelism and reproducibility

Most of the calculations in this guide are sweeps: over k-points for a band
structure, over energies for a density of states, over q-points for a response
function, over restarts for an anneal. They run serially by default, one item
after the other, and `parallel.set_cores(n)` spreads them over `n` worker
processes; `parallel.set_enabled(False)` goes the other way and forces the
whole package strictly serial, which is the setting to reach for when
debugging.

```python
from pyqula import parallel
parallel.set_cores(4)          # spread the sweeps over 4 processes
```

The number of cores changes how long a sweep takes and nothing else. Results do
not depend on the core count, including for the routines that draw random
numbers, the stochastic trace of the KPM and the multistart anneals of the
classical models: `cores=1` and `cores=8` give identical answers. Seeding numpy
in the parent process

```python
import numpy as np
np.random.seed(42)
```

therefore makes a whole sweep reproducible run to run, which is what you want
when comparing two calculations that differ in a single parameter.
