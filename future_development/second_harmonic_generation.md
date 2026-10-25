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
  is not built. Multilayers of the three-band model are
  `specialhamiltonian.three_band_tmdc_multilayer()`, with an interlayer hopping t_perp between the
  d_z2 orbitals of the three nearest metal atoms only (the maintainer's choice, 23 October 2026),
  one value per interface,
  which splits the Gamma valence band of a homobilayer by 6|t_perp| and cancels exactly at K and
  K'. With it, at t_perp = 0.1 eV and nk=60, the static |chi_yyy| of MoS2 is 0.585 nm^2/V for the
  monolayer, zero for the 2H bilayer, 1.187 nm^2/V for the R bilayer, and only 0.002 nm^2/V for the
  2H bilayer with a 0.1 eV step between the layers: since the valleys are not coupled, the step
  alone hardly gives an in-plane response. For the tetralayer question (hybridization against
  the built-in field) this means the in-plane answer depends on the valley coupling, which the
  d_z2-only t_perp leaves out. `interlayer="slater_koster"` (V_sigma,V_pi,V_delta) now couples the
  valleys: with (-0.3,0.1,0.02) eV the K valence band of the 2H MoS2 bilayer (spinless) splits by
  6.6 meV and the 0.1 eV step gives 0.0084 nm^2/V, still small, so whether the step matters in the
  heterostructure depends on integrals that are not fitted. The eleven-band model, with the
  chalcogen orbitals that mediate the interlayer coupling, is the controlled alternative. t_perp itself has no fitted value: it should be
  chosen from the first-principles Gamma splitting of the bilayer. A heterostructure shares the
  average lattice constant and needs its band alignment set by hand through `onsite`.
- Excitonic SHG is built (23 October 2026, `conductivitytk/excitonicoptics.py`): the length-gauge
  sum over Tamm-Dancoff excitons of Ruan, Chan and Louie (arXiv:2310.09674, Eq. (1)), done through the
  resolvent of the BSE matrix, diagonalized once. The position operator between two excitons is
  applied to the exciton written as an orbital matrix Phi_k = sum A_cvk |u_ck><u_vk|, which is free of
  the Bloch phases and periodic in k, as i dPhi/dK + [tau,Phi] with a spectral k-derivative, so no
  smooth gauge is needed (the step ab initio codes do with a gauge construction). An independent
  derivation by a second agent gave the same prefactor q^3/(2 N_k V_cell), the same three terms and
  2w -> 2w + 2i eta. Measured: the kernel="none" limit reproduces `shg.py`, to 1e-4 below the
  resonances at nk=24 and on a resonance (w=2.2, eta=0.1) to 20%, 2% and 0.3% at nk=24, 36, 48; the
  result is invariant under the eigenvector gauge (4e-14); with a Coulomb tail on the gapped
  honeycomb lattice the D3h relation xyy = -xxx holds to 1e-4 of the signal at nk=21 below the gap
  (the bound exciton spans three or four mesh points), and the exciton at 1.845 t gives a peak at
  w = 0.92 t four times the response without interaction. The linear version misses, through
  J = dP/dt, the frequency-independent Hall term; it is added at the independent-particle level,
  which makes the zero-interaction limit exact for every component, including the Haldane model.
- The position operator of the excitonic SHG was rebuilt on 25 October 2026. The spectral
  derivative in reduced coordinates broke C3 in a stacked multilayer: the layers at a third of the
  cell map onto themselves only up to a lattice vector and a k-dependent Bloch phase, so no Fourier
  box in reduced coordinates is covariant (square box 0.91, hexagonal box 0.33, Marzari-Vanderbilt
  differences 0.45 to 0.56, as the C3 violation of the operator on a three-band tetralayer at nk=24).
  The operator now multiplies the real-space pair amplitude phi_ij(R) by the bond vector
  -(R + tau_j - tau_i), shortest image on the supercell, ties averaged (1e-13). On that tetralayer at
  nk=24, delta=0.025, nc=4, the forbidden components went from 0.8 to 1e-12 of yyy. The second
  defect is resolution and is not fixed by any operator: the resonant pair amplitude has width
  delta/v in k, so convergence needs delta >= v dk/2 and every conduction band (the velocity gauge
  itself is 14% off at delta=0.025, nk=24). Against the velocity gauge at nk=120 with nc=8, the
  kernel="none" limit is within 2.7% of yyy at nk=24, delta=0.1, 4.1% under 1% uniaxial strain,
  and 8% (mean 4%) at nk=48, delta=0.05 strained. Left open: an all-band window at nk>24 is beyond
  the dense BSE (nc=8 at nk=24 is about 22 GB), and kernel="none" still diagonalizes a diagonal
  matrix; the commutator form X R(z) d = R(z) X d + i R(z) V R(z) d was measured as an exact-C3
  kernel="none" cross-check but is worse under strain, so it was not built.
  The second-order zero-interaction limit holds without time reversal too: with a Haldane term
  (gap 1.55) it agrees with `shg.py` to 1e-4 of the largest component at nk=24 below the half gap,
  and with a smaller gap (0.65) at nk=48, where at nk=24 even `shg.py` is not converged. With the
  interaction on there is no external reference: every check is internal (zero-interaction limit,
  gauge invariance, lattice symmetry) or qualitative (the 2w enhancement at the bright exciton, as
  in Ruan et al.).
  Cost for spinful MoS2 (three-band, eight pairs per k-point): 3 s at nk=12, 23 s at 18, 117 s at 24;
  nk=30 needs 3.3 GB. Open: the antiresonant terms use the resonant (Tamm-Dancoff) excitons, as in
  both papers, and are not checked against a full BSE; the iterative and quantics BSE solvers, which
  would reach finer meshes, return only a few excitons and are not wired in; a truncation to the
  excitons in an energy window is not offered.
- Three-dimensional Hamiltonians are rejected, as in the linear Kubo code, because
  `kubo._setup` has no 3D branch.
- The length gauge leaves out the Fermi-surface terms, so it is a check for cold insulators only;
  the velocity gauge includes them, but a metal has not been checked against an independent
  reference.
- Only SHG, chi(2w;w,w), is exposed. The recursion gives sum- and difference-frequency generation,
  and the shift and injection currents at w1 = -w2, with the same kernel and two different
  frequencies; the broadening convention at w1 + w2 -> 0 then needs care (Passos et al., the
  section on relaxation parameters).
