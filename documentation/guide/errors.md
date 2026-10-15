# Errors and unsupported inputs

Many calculations only make sense for some Hamiltonians: a Berry curvature
needs a two-dimensional Hamiltonian, a spin operator needs the spin degree of
freedom, a superconducting quantity needs the Nambu one, and a keyword such
as `mode=`, `solver=` or `channel=` accepts only some strings. pyqula checks
these conditions before the calculation starts, and the error message says
what is needed:

- `ValueError`: a value that cannot work, for example an unknown `mode`, or a
  spin operator on a spinless Hamiltonian.
- `NotImplementedError`: a combination that is not available yet.
- `TypeError`: an argument of the wrong type.

For a keyword given as a string, the message lists the accepted values, so a
typo is easy to find:

```
ValueError: unknown mode ED2; the DOS accepts 'ED', 'KPM', 'adaptive', 'Green' and 'RG'
```

Two messages tell you directly what to do. A superconducting quantity on a
normal Hamiltonian asks you to call `h.setup_nambu_spinor()` first, and a
spin quantity on a spinless Hamiltonian asks you to call `h.turn_spinful()`:

```python
from pyqula import geometry
h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
h.extract("mz")
# ValueError: the magnetization 'mz' needs a spinful Hamiltonian; call
# h.turn_spinful() first, or build it with g.get_hamiltonian(has_spin=True)
```

The check is only on reading a degree of freedom that the Hamiltonian does
not have. The methods that add a spin term, such as `h.add_exchange()`,
`h.add_zeeman()` and `h.add_kane_mele()`, make a spinless Hamiltonian
spinful instead of raising an error (see
[Building a Hamiltonian](hamiltonian.md)). In the same way, quantities
resolved by sublattice need a unit cell with sublattice labels, and
`h.extract("CDW")` on a cell without them raises an error instead of
returning `None`.

Some other checks exist because the calculation could run, but the number it
gives would have no meaning:

- `h.get_average_spin_splitting()`, `h.get_spin_splitting_density()` and `h.get_spin_splitting_vs_energy()` need $S_z$ to be conserved, and raise an error for a Hamiltonian with spin-orbit coupling or non-collinear order, giving the largest element that mixes the spins.
- `h.get_dos()` of a non-Hermitian Hamiltonian only works with `mode="ED"`, since the other modes assume real energies (see [Non-Hermitian Hamiltonians](non-hermitian.md)).
- `h.get_total_energy(fermi=...)` does not accept a Nambu Hamiltonian, because there a Fermi energy moves the electron and hole bands in opposite directions; use `h.shift_fermi(-mu)` on the Hamiltonian instead.
- The surface density of states does not accept an operator that depends on momentum, such as `"valley"`, since the momentum is already integrated.
- The lead Green's functions behind `HT.didv()`, `h.get_dos(mode="RG")` and `Embedding.get_gf()` raise an error for an energy exactly on a level of the lead with a broadening below about `1e-7`; use a larger `delta`.
