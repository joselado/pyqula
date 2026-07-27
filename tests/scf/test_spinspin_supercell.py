import numpy as np

from pyqula import geometry
from pyqula import meanfield

MAXERROR = 1e-6


def test_szsz_supercell_matches_minimal_cell():
    """The self-consistent energy per atom (and magnetization pattern) of
    an SzSz calculation must be the same whether computed in the minimal
    (1-atom) cell or an N-times supercell of it -- the physics is the same
    lattice, just described with a bigger repeated unit. The two k-meshes
    must be scaled consistently (nk_supercell = nk_minimal/N, so both
    sample the same physical k-points folded into the smaller BZ) for
    exact agreement -- an under-resolved or mismatched supercell k-mesh
    shows up as ordinary k-convergence noise, not a bug."""
    g = geometry.chain()
    nk0 = 60

    h1 = g.get_hamiltonian(has_spin=True)
    scf1 = meanfield.SzSz(h1, J1=-2.0, mf="ferroZ", nk=nk0, maxerror=MAXERROR,
            mix=0.3, filling=0.2)
    assert scf1.converged
    m1 = scf1.hamiltonian.get_magnetization()

    for n in (2, 3):
        gn = g.get_supercell(n)
        hn = gn.get_hamiltonian(has_spin=True)
        scfn = meanfield.SzSz(hn, J1=-2.0, mf="ferroZ", nk=nk0//n,
                maxerror=MAXERROR, mix=0.3, filling=0.2)
        assert scfn.converged
        assert np.isclose(scf1.total_energy, scfn.total_energy/n, atol=1e-5), \
            (n, scf1.total_energy, scfn.total_energy/n)
        mn = scfn.hamiltonian.get_magnetization()
        assert np.allclose(mn, np.tile(m1, (n, 1)), atol=1e-4), (n, m1, mn)


def test_jinteraction_supercell_matches_minimal_cell():
    """Same check as SzSz, for the combined isotropic-exchange SCF loop
    (Jinteraction), which -- unlike SzSz -- rotates the density matrix
    per-iteration rather than the whole Hamiltonian, so this exercises a
    structurally different code path."""
    g = geometry.chain()
    nk0 = 40

    h1 = g.get_hamiltonian(has_spin=True)
    scf1 = meanfield.Jinteraction(h1, Jx1=-2.0, Jy1=-2.0, Jz1=-2.0,
            mf="ferroZ", nk=nk0, maxerror=MAXERROR, mix=0.2, maxite=300,
            filling=0.2)
    assert scf1.converged

    g2 = g.get_supercell(2)
    h2 = g2.get_hamiltonian(has_spin=True)
    scf2 = meanfield.Jinteraction(h2, Jx1=-2.0, Jy1=-2.0, Jz1=-2.0,
            mf="ferroZ", nk=nk0//2, maxerror=MAXERROR, mix=0.2, maxite=300,
            filling=0.2)
    assert scf2.converged
    assert np.isclose(scf1.total_energy, scf2.total_energy/2, atol=1e-5), \
        (scf1.total_energy, scf2.total_energy/2)


def test_vjinteraction_nambu_supercell_consistency():
    """Same check as the above, for VJinteraction on a BdG (Nambu)
    Hamiltonian combining an attractive V1 (superconductivity) with a
    ferromagnetic J1: both the gap and the total energy per atom must
    match between the primitive (2-site) bichain cell and its 2x (4-site)
    supercell.

    This is a regression test for a real bug (found via this exact check,
    and fixed both here and in the shared densitydensity.py code
    Vinteraction itself uses) in the total-energy computation:
    get_dc_energy(v, dm) assumes dm's shape matches v's (2n, never
    Nambu-doubled), but the total energy used to be computed with the
    full, un-extracted Nambu-sized density matrix for a BdG Hamiltonian --
    which was NOT supercell-consistent (verified directly: with that bug,
    energy per atom differed between cell sizes by an amount much larger
    than the SCF tolerance)."""
    g = geometry.bichain()
    nk0 = 40

    def run(gg, nk):
        h = gg.get_hamiltonian()
        h.add_exchange([0.3, 0., 0.])
        h.turn_nambu()
        return meanfield.VJinteraction(h, V1=-2.0, J1=-1.0, filling=0.3,
                mf="random", nk=nk, maxerror=MAXERROR, mix=0.15, maxite=1000)

    scf1 = run(g, nk0)
    assert scf1.converged
    natoms1 = len(g.r)

    g2 = g.get_supercell(2)
    scf2 = run(g2, nk0//2)
    assert scf2.converged
    natoms2 = len(g2.r)
    assert natoms2 == 2*natoms1

    assert np.isclose(scf1.hamiltonian.get_gap(), scf2.hamiltonian.get_gap(),
            atol=1e-3)
    assert np.isclose(scf1.total_energy/natoms1, scf2.total_energy/natoms2,
            atol=1e-3), (scf1.total_energy/natoms1, scf2.total_energy/natoms2)
