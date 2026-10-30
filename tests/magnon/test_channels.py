"""The spin-flip restriction of the pair basis, and what it may not change.

Restricting the electron-hole pair basis to the magnon block is an
optimization -- it cuts the matrix dimension by about two and the dense
solve by about eight -- so every energy it returns must also be in the
spectrum of the unrestricted problem. It is also the part of the module
easiest to get subtly wrong, because the two halves of the Casida matrix
need DIFFERENT subsets of the pairs (see spinflip_masks), so it is checked
against the unrestricted answer on three states with different spin
structure rather than on one.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.bsetk import spinflip
from pyqula.meanfield import VJinteraction

NK = 6


def _assert_subset(h, Q, tol=1e-10):
    """Every restricted magnon energy must appear in the unrestricted
    spectrum, and there must be fewer of them"""
    a = np.sort(h.get_magnon_energies(nk=NK, Q=Q, channel="auto").real)
    b = np.sort(h.get_magnon_energies(nk=NK, Q=Q, channel="all").real)
    assert len(a) <= len(b)
    worst = max(np.min(np.abs(b - x)) for x in a)
    assert worst < tol, f"restricted energy missing from the full spectrum: {worst}"
    return a, b


def _neel_honeycomb():
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    return h.get_mean_field_hamiltonian(U=3.0, filling=0.5, mf="antiferro",
                                        nk=NK, maxerror=1e-10)


def test_restricted_spectrum_is_a_subset_for_an_antiferromagnet():
    a, b = _assert_subset(_neel_honeycomb(), [0.1, 0., 0.])
    assert 4*len(a) == len(b)  # one of the four spin sectors of the pair basis


def test_restricted_spectrum_is_a_subset_with_a_neighbor_shell_interaction():
    g = geometry.honeycomb_lattice()
    scf = VJinteraction(g.get_hamiltonian(), U=3.0, V1=0.5, filling=0.5,
                         mf="antiferro", nk=NK, maxerror=1e-10, mix=0.3,
                         maxite=2000)
    _assert_subset(scf.hamiltonian, [0.1, 0., 0.])


def test_restricted_and_full_agree_for_a_saturated_ferromagnet():
    """The case that catches a wrong branch selection. Here the two halves
    of the Casida matrix are not mirror images of each other -- the
    opposite spin channel is empty -- so the excitation and de-excitation
    branches interleave instead of separating at zero, and picking "the
    highest half of the spectrum" silently returns the wrong sign. The
    conserved norm ||X||^2-||Y||^2 is what separates them; this state has
    genuinely negative magnon energies (it is an unstable saddle) so the
    sign is not decoration."""
    g = geometry.chain()
    h = g.get_hamiltonian()
    mf = h.copy()
    mf.add_exchange([0., 0., 3.0])
    hmf = h.get_mean_field_hamiltonian(U=10.0, filling=0.5, mf=mf, nk=NK,
                                       maxerror=1e-10)
    a, b = _assert_subset(hmf, [0.1, 0., 0.])
    assert len(a) == len(b)  # only one spin sector exists here
    assert np.min(a) < -1e-3  # unstable against a finite-Q spin wave


def test_spinflip_channel_is_refused_for_a_non_collinear_state():
    h = _neel_honeycomb()
    tilted = h.copy()
    tilted.global_spin_rotation(vector=[1., 0., 0.], angle=0.5)
    with pytest.raises(ValueError):
        spinflip.magnon_matrix(tilted, Q=[0., 0., 0.], nk=NK,
                               channel="spinflip")
    # ... while "auto" quietly keeps the whole basis instead of failing
    p = spinflip.magnon_matrix(tilted, Q=[0., 0., 0.], nk=NK, channel="auto")
    assert p.n1 == p.pb.npair


def test_degenerate_multiplets_do_not_hide_the_spin_flip_block():
    """A magnetic band structure has accidental degeneracies between an up
    and a down band, where the eigensolver returns arbitrary mixtures with
    no Sz character at all. Without rotating those multiplets into Sz
    eigenstates first, the restriction silently gives up and falls back to
    the whole basis -- correct, but eight times slower. nk=6 hits such a
    degeneracy on this state and nk=4 does not, which is exactly how
    subtle the effect is."""
    h = _neel_honeycomb()
    for nk in (4, 6):
        p = spinflip.magnon_matrix(h, Q=[0., 0., 0.], nk=nk)
        assert p.n1 == p.pb.npair//4, f"no spin-flip block found at nk={nk}"


def _crossing_spin_bands():
    """A gapped ionic chain whose two spin species hop differently and are
    split by a Zeeman term, so that the lowest band is up at k=0 and down
    at k=pi. The band index then does not carry the same spin at k and at
    k+Q, which a Neel or ferromagnetic test state never shows: there the
    spin of band n is the same everywhere in the Brillouin zone"""
    g = geometry.chain().supercell(2)
    h = g.get_hamiltonian()
    hop, onsite, zeeman = {0: 1.0, 1: 0.3}, [1.0, -1.0], 0.3
    h.intra = np.zeros((4, 4), dtype=np.complex128)
    h.inter = np.zeros((4, 4), dtype=np.complex128)
    for s in (0, 1): # up, down
        for i in (0, 1):
            h.intra[2*i+s, 2*i+s] = onsite[i] + zeeman*(1-2*s)
        h.intra[s, 2+s] = h.intra[2+s, s] = hop[s]
        h.inter[2+s, s] = hop[s] # site 1 to site 0 of the next cell
    return h


def test_restricted_spectrum_is_a_subset_when_band_spins_cross():
    """The de-excitation half of the Casida matrix needs the pairs whose
    OWN spin change is the opposite of the excitation one. That is the
    same set as reading it off the excitation pair only when every band
    keeps its spin between k and k+Q, so this case is what separates
    them; with the wrong set the restricted spectrum missed the exact one
    by 3e-4 here at Q=0.25"""
    from pyqula.bsetk.interaction import density_interaction
    h = _crossing_spin_bands()
    V = density_interaction(h, U=1.0)
    for Q in ([0.25, 0., 0.], [0.4, 0., 0.]):
        a = np.sort(h.get_magnon_energies(nk=8, Q=Q, V=V, channel="auto").real)
        b = np.sort(h.get_magnon_energies(nk=8, Q=Q, V=V, channel="all").real)
        assert len(a) < len(b)
        worst = max(np.min(np.abs(b - x)) for x in a)
        assert worst < 1e-10, f"restricted energy missing at Q={Q}: {worst}"


def test_magnon_weights_are_residues_in_the_metric():
    """weights[i] is the residue |x_i^+ S v|^2 of the spin generator v in
    mode i, with x_i normalized in the Casida metric, x^+ S x = 1, and the
    weights summed to one. With the Euclidean norm instead, the modes with
    a large de-excitation part came out underweighted, the Goldstone mode
    first (0.81 against 0.93 here)"""
    from pyqula.bsetk.solve import solve_pseudo_hermitian
    h = _neel_honeycomb()
    Q = [0.05, 0., 0.]
    es, w = spinflip.magnon_spectrum(h, Q=Q, nk=NK)
    p = spinflip.magnon_matrix(h, Q=Q, nk=NK)
    e, x = solve_pseudo_hermitian(p.M)
    S = np.ones(p.M.shape[0])
    S[p.n1:] = -1.
    norm = np.real(np.einsum("ia,a,ia->i", x.conj(), S, x))
    keep = norm > 0. # the excitation branch
    e, x = e[keep], x[keep]/np.sqrt(norm[keep])[:, None]
    ref = np.abs(x.conj() @ (S*spinflip.goldstone_vector(p)))**2
    ref = ref/np.sum(ref)
    order = np.argsort(e.real)
    assert np.allclose(es, e[order])
    assert np.allclose(w, ref[order], atol=1e-8)


def test_casida_solve_takes_halves_of_different_sizes():
    """A magnon problem has an excitation half and a de-excitation half of
    different sizes as soon as the magnet is not an antiferromagnet. The
    metric of solve_pseudo_hermitian has to follow that split, or the
    Cholesky of S@H fails every time and a matrix of odd size cannot be
    solved at all. Checked on stable random Casida matrices against the
    general eigensolver"""
    import scipy.linalg as lg
    from pyqula.bsetk.solve import solve_pseudo_hermitian
    rng = np.random.default_rng(3)
    def herm(n):
        x = rng.normal(size=(n, n)) + 1j*rng.normal(size=(n, n))
        return (x + x.conj().T)/10. + 3.*np.identity(n)
    for (n1, n2) in [(8, 4), (4, 8), (10, 3)]:
        b = 0.2*(rng.normal(size=(n1, n2)) + 1j*rng.normal(size=(n1, n2)))
        M = np.block([[herm(n1), b], [-b.conj().T, -np.conj(herm(n2))]])
        es, ws = solve_pseudo_hermitian(M, n1=n1)
        ref = lg.eig(M)[0]
        assert np.max(np.abs(es.imag)) == 0. # the Hermitian route ran
        assert np.max(np.abs(np.sort(es.real) - np.sort(ref.real))) < 1e-12
        assert np.max(np.abs(M@ws.T - ws.T*es[None, :])) < 1e-12
        # the positive branch is the upper n1 of the spectrum
        ep, _ = solve_pseudo_hermitian(M, n1=n1, positive=True)
        assert len(ep) == n1 and np.min(ep.real) > 0.


def test_magnons_of_a_ferrimagnet_with_an_odd_number_of_pairs():
    """The Lieb lattice at half filling is a ferrimagnet, with one
    spin-flip pair per k-point of one kind against four of the other, so
    at nk=5 the Casida matrix is 25 + 100 = 125 wide. It used to fail with
    a broadcasting error"""
    g = geometry.lieb_lattice()
    scf = VJinteraction(g.get_hamiltonian(), filling=0.5, mf="ferro", nk=5,
                        maxerror=1e-10, mix=0.3, maxite=3000, U=3.0,
                        verbose=0)
    h = scf.hamiltonian
    Q = [0.2, 0.1, 0.]
    p = spinflip.magnon_matrix(h, Q=Q, nk=5)
    assert (p.n1 + p.n2) % 2 == 1
    es = np.sort(h.get_magnon_energies(nk=5, Q=Q).real)
    ev, vr = np.linalg.eig(p.M) # the general route, by the metric
    vr = vr.T/np.linalg.norm(vr.T, axis=1)[:, None]
    met = (np.sum(np.abs(vr[:, :p.n1])**2, axis=1)
           - np.sum(np.abs(vr[:, p.n1:])**2, axis=1))
    ref = np.sort(ev[np.argsort(-met)[:p.n1]].real)
    assert np.max(np.abs(es - ref)) < 1e-10


def test_a_ferrimagnet_has_the_same_magnons_polarized_up_or_down():
    """The same Lieb ferrimagnet, and its exact image under a spin flip
    (spin indices swapped in every hopping and in h.V): the magnons are
    the same, Goldstone zero included. Choosing the S^- channel whenever
    it had a pair kept the gapped branch of the down-polarized state
    (0.906 at Q=0), since there the acoustic mode has negative norm"""
    g = geometry.lieb_lattice()
    scf = VJinteraction(g.get_hamiltonian(), filling=0.5, mf="ferro",
                        nk=NK, maxerror=1e-10, mix=0.3, maxite=3000, U=3.0,
                        verbose=0)
    h1 = scf.hamiltonian.get_multicell()
    n = h1.intra.shape[0]
    P = np.zeros((n, n))
    for i in range(n//2): P[2*i, 2*i+1] = P[2*i+1, 2*i] = 1.
    h2 = h1.copy()
    h2.intra = P@h1.intra@P.T
    for t in h2.hopping: t.m = P@t.m@P.T
    h2.V = {k: P@np.array(m)@P.T for k, m in h1.V.items()}
    assert np.sum(h1.get_vev("sz"))*np.sum(h2.get_vev("sz")) < 0.
    for Q in [[0., 0., 0.], [0.2, 0.1, 0.]]:
        e1 = np.sort(h1.get_magnon_energies(nk=NK, Q=Q).real)
        e2 = np.sort(h2.get_magnon_energies(nk=NK, Q=Q).real)
        assert len(e1) == len(e2)
        assert np.max(np.abs(e1 - e2)) < 1e-8
    e0 = np.sort(h1.get_magnon_energies(nk=NK, Q=[0., 0., 0.]).real)
    assert abs(e0[0]) < 1e-6 # the Goldstone mode is kept
