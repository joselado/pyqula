---
name: error-conventions
description: How pyqula raises errors -- which exception type for which failure, the registries behind string-selected options (mode=, solver=, channel=, operator names), and the shared Hilbert-space guards in check.py. Load this before writing any raise statement in src/pyqula, before adding or changing an option selected by a string, and before writing a guard that checks whether a Hamiltonian is spinful, has Nambu or has a sublattice -- the package went through a deliberate sweep to reach this state and a hand-written guard undoes part of it. Also load it when reviewing a diff that adds error handling, or when a test asserts on an error message.
---

# Error conventions

Argument and Hilbert-space guards raise a real exception with a message saying what the
routine requires: `ValueError` for a bad input value or a Hamiltonian in the wrong
Hilbert space, `NotImplementedError` for a combination that is simply not built yet,
`TypeError` for a wrong type.

## The two forms that were swept out

Neither should come back:

- a bare `raise`, which surfaces as `RuntimeError: No active exception to reraise` and
  names neither the input nor the requirement
- a bare `raise NotImplementedError`, which names the category but not what is unsupported

Where a guard's message can name the offending value -- the mode string, the two
mismatched sizes, the type that was passed -- it does, rather than printing it and
raising separately.

### The expected bare-`raise` count is nine

The only bare `raise` left in `src/pyqula` is the jump-to-except idiom inside a `try`
body (five sites), where it is control flow rather than an error report, plus four
ordinary re-raises inside `except` handlers, which re-raise a live exception and are not
the message-less form at all. An AST walk therefore finds nine bare `raise` nodes in the
package. Nine is the expected count, not a regression:

```bash
python -c "
import ast,os
n=0
for root,d,fs in os.walk('src/pyqula'):
    for f in fs:
        if not f.endswith('.py'): continue
        for node in ast.walk(ast.parse(open(os.path.join(root,f)).read())):
            if isinstance(node,ast.Raise) and node.exc is None: n+=1
print(n)"
```

## Options selected by a string

An option selected by a string (`mode=`, `solver=`, `channel=`, an operator name) lists
the accepted values in the error, so that a typo is self-diagnosing.

Five dispatches go one step further and keep their names in a registry, so the accepted
set can be enumerated and adding a name is one dict entry rather than a new `elif`
branch. What matters is that the advertised list is *derived from* the registry instead
of being a second list maintained by hand beside the chain -- that is what stops the two
from drifting, which is the failure this shape exists to prevent:

| dispatch | registry | names |
| --- | --- | --- |
| named operators | `operatorlist.py` | `operatorlist.get_operator_names()` |
| mean-field guesses (`mf=`) | `meanfield.py` | `meanfield.get_guess_names()` |
| pairing symmetries (`mode=`) | `sctk/pairing.py` | `pairing.get_pairing_modes()` |
| high-symmetry kpoint labels | `kpointstk/labels.py` | `labels.get_label_names()` |
| `h.extract(name)` quantities | `extract.py` | `extract.get_extractable_names()` |

A new string-selected option should follow that shape rather than adding an `elif`.

## Hilbert-space requirements

These go through three shared guards in `check.py`:

- `require_spin(h,what)`
- `require_nambu(h,what)`
- `require_sublattice(h,what)`

where `what` is the noun phrase the message opens with (`"an exchange field"`,
`"the d-vector"`). They supply the fixed tail naming the remedy (`h.turn_spinful()`,
`h.setup_nambu_spinor()`), so a new routine does not reinvent either the wording or the
fix. Use them instead of a hand-written
`raise ValueError("... needs a spinful Hamiltonian")`.

Two families of guard carry *more* information than the generic one and are deliberately
left as they are: the `check_mode("spinful_nambu")` family, which names both flags'
values, and the local Hubbard-U guard, which points at the spinless `V1/V2/V3/Vr`
alternative. When a specific message says more than the shared one would, keep it.
