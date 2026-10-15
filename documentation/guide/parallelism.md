# Parallelism and reproducibility

Many calculations are loops over independent items: k-points for a band
structure, energies for a density of states, q-points for a response
function, restarts for an annealing. By default they run one after the
other. `parallel.set_cores(n)` runs them on `n` processes:

```python
from pyqula import parallel
parallel.set_cores(4)          # spread the sweeps over 4 processes
```

The number of cores changes only the time of the calculation, not the
result. This is also true for the calculations that use random numbers, such
as the stochastic trace of the [kernel polynomial method](kpm.md) or the
restarts of the classical models: `cores=1` and `cores=8` give the same
numbers. So if you fix the random seed of numpy in your script,

```python
import numpy as np
np.random.seed(42)
```

the whole calculation gives the same result every time. This is useful when
you compare two calculations that differ in one parameter only.

`parallel.set_enabled(False)` makes everything run on one core, with no
parallel processes and no threads. Use it to debug a calculation, or to
measure its time on a single core. `parallel.set_enabled(True)` allows
parallel runs again, but it does not restore the previous number of cores,
so call `parallel.set_cores(n)` again.
