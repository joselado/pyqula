# Errors and unsupported inputs

Most routines only make sense for a Hamiltonian of a particular kind: a Berry
curvature needs a two-dimensional one, a spin operator needs the spin degree
of freedom, a superconducting quantity needs the Nambu one, and an option
selected by a string (`mode=`, `solver=`, `channel=`) needs one of a fixed set
of strings. Those requirements are checked up front, before any calculation
starts, and the exception says what was needed rather than only that something
went wrong: `ValueError` for a value that cannot work (the Berry curvature of a
Hamiltonian that is not two-dimensional, an unknown `mode` string, a spin
operator on a spinless Hamiltonian), `NotImplementedError` for a combination
that is not built yet, `TypeError` for an argument of the wrong kind. For a
string-selected option the accepted values are listed in the message, so a
typo is self-diagnosing:

```
ValueError: unknown mode ED2; the DOS accepts 'ED', 'KPM', 'adaptive', 'Green' and 'RG'
```

Two messages name their own fix, and you will meet them: a superconducting
quantity asked of a normal Hamiltonian says to call `h.setup_nambu_spinor()`
first, and a spin quantity asked of a spinless one says to call
`h.turn_spinful()`.

```python
from pyqula import geometry
h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
h.extract("mz")
# ValueError: the magnetization 'mz' needs a spinful Hamiltonian; call
# h.turn_spinful() first, or build it with g.get_hamiltonian(has_spin=True)
```

Note that the requirement is on *reading* a degree of freedom the Hamiltonian
does not have. The methods that *add* a spin term, `h.add_exchange()`,
`h.add_zeeman()`, `h.add_kane_mele()`, promote a spinless Hamiltonian to a
spinful one rather than rejecting it, as we saw in the section on the Zeeman
field. A third requirement of the same kind is the sublattice label, which
several sublattice-resolved quantities need and an unlabeled cell does not
have: `h.extract("CDW")` on such a cell says so rather than returning `None`.

A second family of checks exists not because the routine cannot run, but
because it could run and return a number that means nothing. Those raise
rather than answer:

- `h.get_average_spin_splitting()`, `h.get_spin_splitting_density()` and
  `h.get_spin_splitting_vs_energy()` refuse a Hamiltonian whose spin
  off-diagonal block does not vanish (Rashba, any spin-orbit term,
  non-collinear order), naming the largest off-diagonal element found
- `h.get_dos()` on a non-Hermitian Hamiltonian refuses every mode but `"ED"`,
  and refuses `use_kpm=True`: the Chebyshev and adaptive expansions assume a
  real spectrum (see "Non-Hermitian Hamiltonians")
- `h.get_total_energy(fermi=...)` refuses a Nambu Hamiltonian, where a nonzero
  `fermi` is not a rigid shift of the spectrum (the electron and hole blocks
  move opposite ways). Use `h.shift_fermi(-mu)` on the Hamiltonian instead
- the surface density of states refuses a momentum-dependent operator such as
  `"valley"`, which its Green's function has already integrated over
- the lead decimation behind `HT.didv()`, `h.get_dos(mode="RG")` and
  `Embedding.get_gf()` refuses an energy sitting exactly on a lead level with
  a broadening below about `1e-7`, and recommends a larger `delta`
