# Orbital magnetic field on a Hamiltonian that already carries pairing

## What exists today

`peierls.add_bfield` (reached from `h.add_peierls` /
`h.add_orbital_magnetic_field`) attaches a Peierls phase
$t_{ij}\to t_{ij}e^{i b\,\phi(r,\delta r)}$ to every hopping. It now works
for spinless, spinful, and Nambu Hamiltonians:

- the site index of an orbital is derived from `dim(H)/nsites` rather than
  assumed to be 2, so the four-orbitals-per-site Nambu layout no longer
  indexes off the end of the position list (it used to die with an
  `IndexError`);
- the hole block picks up the **conjugate** phase, since it carries the
  opposite charge. The acceptance test for this is
  `tests/superconductivity/test_nambu_combinations.py`: adding the field
  before `turn_nambu()` and adding it after must give the same spectrum,
  and they agree to machine precision.

## What is deliberately not built

If the Hamiltonian already carries a **nonzero pairing amplitude**,
`add_bfield` raises `NotImplementedError` instead of guessing. There is no
single Peierls phase for the anomalous block: a Cooper pair carries charge
$2e$, so under a vector potential $\Delta(r)$ acquires $e^{2i\int A}$ rather
than the single-particle phase, and in a genuine orbital field a type-II
superconductor responds by nucleating **vortices** — a spatially varying
$\Delta(r)$ with $2\pi$ phase winding, which is a self-consistent
calculation on a supercell, not a phase factor stamped onto an existing
matrix.

The supported workflow, and the one the error message points at, is
therefore: build the normal-state Hamiltonian, add the orbital field,
*then* `turn_nambu()`/`add_swave()`. `superconductivity.build_nambu_matrix`
constructs the hole block as $-\mathcal{T}H\mathcal{T}^{-1}$, so the
conjugate phases appear automatically and the result is correct.

## What a real implementation would need

1. A supercell large enough to hold an integer number of flux quanta (the
   same commensuration constraint as the existing Hofstadter/Peierls work
   in `tests/bandstructure/test_aah_model_hofstadter_pumping.py`).
2. A **self-consistent** $\Delta(r)$ from `get_mean_field_hamiltonian`
   with the field present, seeded with a winding phase — the vortex core
   is where $|\Delta|$ is suppressed, and that suppression is the whole
   physics; a rigid $\Delta$ with a stamped-on phase is not an
   approximation of it.
3. A gauge choice consistent between the normal and anomalous terms, plus
   a check that the answer does not depend on it (the Landau/symmetric
   switch already in `add_bfield` is the natural test).

Worth doing only if vortex-lattice / mixed-state physics is actually
wanted; nothing in the library needs it today, and the workflow above
covers every case where the field is applied to a normal state that is
subsequently paired.
