"""The partly filled shell of a gapless Bogoliubov-de Gennes state
(scftk.superscf.ShellFilling). In a state with no gap on the mesh the
paired charge is a staircase in mu, and the requested charge falls inside
a jump, so no Fermi level holds it. The density matrices of "ed" and of
the triangle then fill the shell of levels at zero energy in part, at a
chemical potential shifted inside the shell, so that the charge is
exactly the requested one. A gapped state has no level in the shell and
does not change."""
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


def _swave(zeeman=0., **kw):
    """The s-wave state from U=-2 at filling 0.3; with a Zeeman field 0.3
    the pairing dies and the state is the normal one, with no gap"""
    np.random.seed(1)
    return Vinteraction(_h0(zeeman), U=-2., filling=FILLING, mf="swave",
            nk=NK, solver="broyden_mixing", maxerror=1e-10, load_mf=False,
            write=False, verbose=0, **kw)


@pytest.fixture(scope="module")
def zeeman():
    return {integration: _swave(0.3, integration=integration)
            for integration in ("ed", "triangle")}


def _shell(h):
    """The shell of the BdG Fermi level of h at the filling, with the
    compressibility of its normal state"""
    fl = superscf.BdGFermiLevel(h, FILLING,
            lambda hh, f: hh.get_fermi4filling(f, nk=NK, T=1e-7))
    fl.shell.kappa = fl.compressibility(h)
    return fl.shell


def _onsite(h):
    return {(0, 0, 0): np.zeros((2, 2))}


def test_the_gapless_state_holds_the_charge_with_ed_and_the_triangle(zeeman):
    """Without the shell the loop ended with the charge error of one
    k-shell, 1.6e-3 per cell, and "ed" and the triangle differed by 8e-7
    in the energy and 6.5e-4 in the Fermi level"""
    n0 = 2*FILLING # electrons per cell
    for scf in zeeman.values():
        assert scf.converged
        assert abs(scf.charge_error) < 1e-10
        assert abs(superscf.get_charge_bdg(scf.dm)-n0) < 1e-10
        assert superscf.pairing_amplitude(scf.hamiltonian) < 1e-8 # normal
    ed, tri = zeeman["ed"], zeeman["triangle"]
    assert abs(ed.total_energy-tri.total_energy) < 1e-10
    assert abs(ed.hamiltonian.fermi-tri.hamiltonian.fermi) < 1e-8


def test_the_band_energy_is_the_one_of_the_filled_density_matrix(zeeman):
    """The reported energy is variational: the band energy is the
    expectation value of H in the filled density matrix,
    (Tr H rho + Tr h_e)/2 per cell, summed over the hoppings of H"""
    h = zeeman["ed"].hamiltonian
    hd = {d: np.array(algebra.todense(m)) for (d, m) in h.get_dict().items()}
    shell = _shell(h)
    dm = get_dm(h, {d: np.zeros((2, 2)) for d in hd}, nk=NK, T=1e-7,
            integration="ed", shell=shell)
    assert shell.fired
    ee = get_eh_sector(hd[(0, 0, 0)], i=0, j=0)
    band = (sum(np.sum(hd[d]*dm[d]) for d in hd).real
            + np.trace(ee).real)/2.
    assert abs(band-(h.get_total_energy(nk=NK)+shell.energy)) < 1e-12


@pytest.mark.parametrize("integration", ["ed", "triangle"])
def test_a_gapped_state_does_not_change(integration, monkeypatch):
    """The s-wave state has no level in the shell: the same iterations,
    energy and density matrix as with the filling switched off"""
    fills = []
    fill = superscf.ShellFilling.fill
    def counted(self, dm):
        out = fill(self, dm)
        fills.append(self.fired)
        return out
    monkeypatch.setattr(superscf.ShellFilling, "fill", counted)
    scf = _swave(integration=integration)
    assert len(fills) > 0 and not any(fills)
    monkeypatch.setattr(superscf.ShellFilling, "fill", lambda self, dm: dm)
    ref = _swave(integration=integration)
    assert scf.iterations == ref.iterations
    assert abs(scf.total_energy-ref.total_energy) < 1e-12
    assert np.max(np.abs(scf.dm[(0, 0, 0)]-ref.dm[(0, 0, 0)])) < 1e-12
    # and the density matrix of the converged state, with and without it
    h = scf.hamiltonian
    shell = _shell(h)
    monkeypatch.setattr(superscf.ShellFilling, "fill", fill)
    a = get_dm(h, _onsite(h), nk=NK, T=1e-7, integration=integration)
    b = get_dm(h, _onsite(h), nk=NK, T=1e-7, integration=integration,
            shell=shell)
    assert len(shell.levels) == 0 and not shell.fired
    assert np.array_equal(a[(0, 0, 0)], b[(0, 0, 0)])


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


@pytest.mark.parametrize("integration", ["ed", "triangle"])
def test_the_filled_density_matrix_is_hermitian_and_basis_free(zeeman,
        integration, monkeypatch):
    """At the converged gapless state a star of 8 k-points sits at zero
    energy, with its electron and hole levels degenerate at each k. The
    filled density matrix is Hermitian, particle-hole symmetric, and the
    same whatever eigenbasis the diagonalization picks inside the shell.
    A uniform filling of the multiplet with f would not be, since the
    multiplet holds the electron level and its hole partner"""
    h = zeeman[integration].hamiltonian
    def filled():
        shell = _shell(h)
        dm = get_dm(h, _onsite(h), nk=NK, T=1e-7, integration=integration,
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
    # the Fermi-Dirac weight at T reads the rounding of the degenerate
    # energies, 1e-16, amplified by 1/T
    assert np.max(np.abs(filled()-dm)) < 1e-10
    assert len(mixed) >= 8 # the star, one multiplet per k-point
