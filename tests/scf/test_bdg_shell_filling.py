"""The partly filled shell of a gapless Bogoliubov-de Gennes state
(scftk.superscf.ShellFilling). In a state with no gap on the mesh the
paired charge is a staircase in mu, and the requested charge falls inside
a jump, so no Fermi level holds it. The density matrices of "ed" and of
the triangle then fill the shell of levels at zero energy in part, at a
chemical potential shifted inside the shell, so that the charge is
exactly the requested one. A state whose shell is not filled, a gapped
one or one whose charge is held already, does not change."""
import numpy as np
import pytest

from pyqula import geometry, algebra
from pyqula.htk import eigenvectors
from pyqula.scftk import superscf
from pyqula.scftk.densitydensity import Vinteraction, get_dm
from pyqula.superconductivity import get_eh_sector

NK = 16
FILLING = 0.3


def _h0(zeeman=0.):
    h = geometry.square_lattice().get_hamiltonian(has_spin=True)
    if zeeman: h.add_zeeman([0., 0., zeeman])
    h.setup_nambu_spinor()
    return h


def _swave(zeeman=0., U=-2., nk=NK, **kw):
    """The s-wave state from U at filling 0.3; with a Zeeman field 0.3 and
    U=-2 the pairing dies and the state is the normal one, with no gap"""
    np.random.seed(1)
    return Vinteraction(_h0(zeeman), U=U, filling=FILLING, mf="swave",
            nk=nk, solver="broyden_mixing", maxerror=1e-10, load_mf=False,
            write=False, verbose=0, **kw)


@pytest.fixture(scope="module")
def zeeman():
    return {(i, T): _swave(0.3, integration=i, T=T)
            for i in ("ed", "triangle") for T in (1e-7, 1e-3)}


def _shell(h):
    """The shell of the BdG Fermi level of h at the filling, with the
    compressibility of its normal state"""
    fl = superscf.BdGFermiLevel(h, FILLING,
            lambda hh, f: hh.get_fermi4filling(f, nk=NK, T=1e-7))
    fl.shell.kappa = fl.compressibility(h)
    return fl.shell


def _onsite(h):
    return {(0, 0, 0): np.zeros((2, 2))}


def _unfilled(scf):
    """The converged Hamiltonian at the Fermi level before the fill, which
    moved it by scf.shell.mu so that it holds the charge by itself"""
    h = scf.hamiltonian.copy()
    if scf.shell.fired: h.shift_fermi(scf.shell.mu)
    return h


@pytest.mark.parametrize("T", [1e-7, 1e-3])
def test_the_gapless_state_holds_the_charge_with_ed_and_the_triangle(zeeman,
        T):
    """Without the shell the loop ended with the charge error of one
    k-shell, 1.6e-3 per cell, and "ed" and the triangle differed by 8e-7
    in the energy and 6.5e-4 in the Fermi level at T=1e-7"""
    n0 = 2*FILLING # electrons per cell
    ed, tri = zeeman[("ed", T)], zeeman[("triangle", T)]
    for scf in (ed, tri):
        assert scf.converged
        assert abs(scf.charge_error) < 1e-10
        assert abs(superscf.get_charge_bdg(scf.dm)-n0) < 1e-10
        assert superscf.pairing_amplitude(scf.hamiltonian) < 1e-8 # normal
        assert scf.shell.fired
        # the Fermi level moved by the fill holds the charge by itself
        dm = get_dm(scf.hamiltonian, _onsite(scf.hamiltonian), nk=NK, T=T,
                integration="ed")
        assert abs(superscf.get_charge_bdg(dm)-n0) < 1e-8
    assert abs(ed.total_energy-tri.total_energy) < 1e-10
    assert abs(ed.hamiltonian.fermi-tri.hamiltonian.fermi) < 1e-8


@pytest.mark.parametrize("T", [1e-7, 1e-3])
def test_the_energy_of_the_fill_is_the_one_of_its_density_matrix(zeeman, T):
    """With the shell filled the band energy is the one of the filled
    density matrix, (Tr H rho + Tr h_e)/2 per cell summed over the hoppings
    of H, at T=1e-3 too, where the Fermi-Dirac weight is not a step on
    many levels: the shell holds all of them"""
    h = _unfilled(zeeman[("ed", T)])
    hd = {d: np.array(algebra.todense(m)) for (d, m) in h.get_dict().items()}
    v = {d: np.zeros((2, 2)) for d in hd}
    ee = get_eh_sector(hd[(0, 0, 0)], i=0, j=0)
    def band(dm):
        return (sum(np.sum(hd[d]*dm[d]) for d in hd).real
                + np.trace(ee).real)/2.
    shell = _shell(h)
    filled = get_dm(h, v, nk=NK, T=T, integration="ed", shell=shell)
    assert shell.fired
    assert abs(band(filled)-(h.get_total_energy(nk=NK)+shell.energy)) < 1e-12


def _no_shell(monkeypatch):
    """Switch the shell off: start says it may never be filled"""
    start = superscf.ShellFilling.start
    def never(self, h, T, nkpoints):
        start(self, h, T, nkpoints)
        return False
    monkeypatch.setattr(superscf.ShellFilling, "start", never)


@pytest.mark.parametrize("case", [
    dict(integration="ed"), dict(integration="triangle"),
    # the pairing lives, Delta=3.1e-3 of the order of T, and the charge
    # misses N0 by 3.5e-3; at T=0 weights this state moved by 2e-10
    dict(integration="triangle", U=-0.5, nk=32, T=1e-3)])
def test_a_state_whose_shell_is_not_filled_does_not_change(case,
        monkeypatch):
    """The same iterations, energy and density matrix, bit for bit, as
    with the shell switched off"""
    fills = []
    fill = superscf.ShellFilling.fill
    def counted(self, dm):
        out = fill(self, dm)
        fills.append(self.fired)
        return out
    monkeypatch.setattr(superscf.ShellFilling, "fill", counted)
    scf = _swave(**case)
    assert not any(fills)
    _no_shell(monkeypatch)
    ref = _swave(**case)
    assert scf.iterations == ref.iterations
    assert scf.total_energy == ref.total_energy
    assert np.array_equal(scf.dm[(0, 0, 0)], ref.dm[(0, 0, 0)])


@pytest.mark.parametrize("integration", ["ed", "triangle"])
def test_a_nambu_chain_that_holds_its_charge_is_the_normal_one(integration):
    """V1=1 on the chain at half filling from a ferromagnetic guess, nk=20:
    the shell at mu is half filled and holds the charge already, and has
    to split for the state to form. With and without Nambu the same state,
    for each integration (the triangle gives a lower one than "ed"). Before
    the fill left a held charge alone, the Nambu triangle went to the state
    of "ed", 8.6e-3 higher"""
    E = []
    for nambu in (False, True):
        h = geometry.chain().get_hamiltonian()
        if nambu: h.setup_nambu_spinor()
        scf = Vinteraction(h, V1=1.0, filling=0.5, nk=20, mf="ferroZ",
                mix=0.3, maxerror=1e-8, maxite=1000, load_mf=False,
                write=False, verbose=0, integration=integration)
        assert scf.converged
        E.append(scf.total_energy)
    assert abs(E[1]-E[0]) < 1e-9


def _rotated(diagonalize, mixed, seed=2):
    """A diagonalization that mixes every degenerate multiplet at zero
    energy with a random unitary, which is as valid an eigenbasis; mixed
    counts the multiplets"""
    rng = np.random.default_rng(seed)
    def rotated(hks, **kw):
        es, U = diagonalize(hks, **kw)
        U = np.array(U)
        for ik in range(len(es)):
            s = np.nonzero(np.abs(es[ik]) < 1e-6)[0]
            if len(s) < 2: continue
            assert np.ptp(es[ik][s]) < 1e-12 # one multiplet
            q, r = np.linalg.qr(rng.normal(size=(len(s), len(s)))
                    + 1j*rng.normal(size=(len(s), len(s))))
            U[ik][:, s] = U[ik][:, s]@q
            mixed.append(len(s))
        return es, U
    return rotated


@pytest.mark.parametrize("T", [1e-7, 0.])
@pytest.mark.parametrize("integration", ["ed", "triangle"])
def test_the_filled_density_matrix_is_hermitian_and_basis_free(zeeman,
        integration, T, monkeypatch):
    """At the converged gapless state a star of 8 k-points sits at zero
    energy, with its electron and hole levels degenerate at each k. The
    filled density matrix is Hermitian, particle-hole symmetric, and the
    same whatever eigenbasis the diagonalization picks inside the shell,
    also at T=0, where the smearing of the shell is a floor of 1e-11 and
    the rounding of the degenerate energies would decide the filling. A
    uniform filling of the multiplet with f would not be particle-hole
    symmetric, since the multiplet holds the electron level and its hole
    partner"""
    h = _unfilled(zeeman[(integration, 1e-7)])
    def filled():
        shell = _shell(h)
        dm = get_dm(h, _onsite(h), nk=NK, T=T, integration=integration,
                shell=shell)[(0, 0, 0)]
        assert shell.fired and 0. < shell.fraction < 1.
        assert abs(superscf.get_charge_bdg({(0, 0, 0): dm})-2*FILLING) < 1e-12
        return dm
    dm = filled()
    assert np.max(np.abs(dm-dm.conj().T)) < 1e-14
    # particle-hole: the hole occupations of a site are 1 minus the
    # electron ones of the other spin, (c_dn^dag, -c_up^dag)
    ee = get_eh_sector(dm, i=0, j=0).diagonal().real
    hh = get_eh_sector(dm, i=1, j=1).diagonal().real
    other = ee.reshape(-1, 2)[:, ::-1].ravel() # the other spin, site by site
    assert np.max(np.abs(hh-(1.-other))) < 1e-12
    mixed = []
    monkeypatch.setattr(eigenvectors, "parallel_diagonalization",
            _rotated(eigenvectors.parallel_diagonalization, mixed))
    assert np.max(np.abs(filled()-dm)) < 1e-12
    assert len(mixed) >= 8 # the star, one multiplet per k-point
