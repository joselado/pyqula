# Second-harmonic generation: what is built and what is left open

The second-harmonic conductivity and susceptibility, `h.get_shg_conductivity()` and
`h.get_shg_susceptibility()` (`conductivitytk/shg.py`), were added on 23 October 2026 at the
request of the tetralayer MoS2/WSe2 project, whose referees ask whether the SHG of a
2H-MoS2/2H-WSe2 heterobilayer of bilayers comes from interlayer hybridization, from the built-in
field of the interface or from strain. This file records what was measured while building it and
what was deliberately left out.

## What was measured

The production path is the velocity-gauge density-matrix recursion of Passos et al.
(arXiv:1712.04924) with the first, second and third Cartesian k-derivatives of the Peierls-coupled
Bloch Hamiltonian, which is exact for a tight-binding model with the position operator diagonal in
the orbitals, regardless of the number of bands. The length-gauge sum over states, with the
generalized derivative r^c_nm;b taken from the exact identity [D^b,h^c] = h^bc, is kept as the
independent check (`gauge="length"`, cold insulators only).

- On a nondegenerate insulator (the gapped honeycomb lattice, a three-sublattice triangular
  lattice) the two gauges agree to 1e-16 on the same k-mesh, already at nk=24, so the agreement is
  an identity per mesh and not a convergence statement.
- Wherever two occupied bands come close (the biased AA' bilayer, Rashba plus exchange on the
  honeycomb lattice) the length gauge converges slowly in nk, since it divides by band energy
  differences, while the velocity gauge is converged at nk=48; at V=0.6 the two meet within 1e-3 of
  a 1e-2 signal at nk=96. This is the reason the velocity gauge is the default, contrary to the
  recommendation of Taghizadeh et al. (arXiv:1710.01300), which is about truncated continuum bases
  and does not apply to a complete tight-binding basis.
- The absolute prefactor and the sign (carrier charge q = -e) are pinned by the static limit of a
  crystal of isolated molecules against the exact (1/2) d2P/dE2, to 1e-6 relative.
- Cost: 24 bands, 100 frequencies, two components, 2.2 s at nk=40 on 14 threads, scaling as
  nk^2 n^3 times the number of frequencies.
- The two-band hBN model (t = 2.33 eV, onsite 3.9 eV, bond 1.45 A, spinful) gives a static sheet
  chi_xxx = 2.6e-3 nm^2/V, converged to 4e-9 relative at nk=100, and the peak of Im chi at half the M
  transition converged to 1% at nk=100 and 2e-4 at nk=200 for delta=0.02 t.
- In the toy AA' bilayer of two gapped honeycomb layers, a pure potential step V between the layers
  gives only out-of-plane components (zxx, zzz, xxz) and the in-plane xxx stays exactly zero, while
  two different layer masses give in-plane xxx and no out-of-plane response. This is a property of
  that bipartite model and not a general statement about the heterostructure.

## What is left open

- The benchmark against Trolle, Seifert and Pedersen (arXiv:1310.0674) is done in order of
  magnitude only. The three-band model of Liu et al. (arXiv:1305.6089), now
  `specialhamiltonian.three_band_tmdc()`, gives for monolayer MoS2 (TNN, GGA, with SOC) a static
  sheet |chi_yyy| = 0.59 nm^2/V against Trolle's 0.3 nm^2/V from an sp3d5 model, with the D3h
  pattern for x along zigzag, and a largest |chi_yyy| of 5.0 nm^2/V at 1.68 eV (nk=120,
  delta=0.025 eV). The positions of the A/2, B/2 and C/2 features have not been compared. The
  eleven-band model of Fang et al. (arXiv:1506.08860), which carries interlayer and strain terms,
  is not built; the three-band model has no interlayer hopping, so a multilayer needs one added
  by hand, between the d orbitals of neighboring metal atoms.
- Excitonic SHG (Taghizadeh and Pedersen, arXiv:1902.09141) on top of `bsetk/` is not built. In a
  TMD the excitons move the resonances by several hundred meV and change their weight, so a
  single-particle chi2 is a statement about symmetry and trends, not about absolute peak positions.
- Three-dimensional Hamiltonians are rejected, as in the linear Kubo code, because
  `kubo._setup` has no 3D branch.
- The length gauge leaves out the Fermi-surface terms, so it is a check for cold insulators only;
  the velocity gauge includes them, but a metal has not been checked against an independent
  reference.
- Only SHG, chi(2w;w,w), is exposed. The recursion gives sum- and difference-frequency generation,
  and the shift and injection currents at w1 = -w2, with the same kernel and two different
  frequencies; the broadening convention at w1 + w2 -> 0 then needs care (Passos et al., the
  section on relaxation parameters).
