from pyqula import geometry
from pyqula.meanfield import VJinteraction


def test_honeycomb_antiferromagnet_converges_to_a_tight_tolerance():
    """The U=3, V1=0.5 honeycomb antiferromagnet of the user guide
    (spin-response.md) at nk=24 converges to maxerror=1e-10, and the
    Goldstone mode is then at zero to that tolerance. With the default
    mix=0.1 the plain-mixing error falls geometrically by about 0.982 per
    iteration (one slow mode of the fixed-point map, at nk=6 to 24), with no
    floor down to 1e-13, so maxite=1000 stopped it near 1e-9; mix=0.5 needs
    about 220 iterations."""
    nk = 24
    g = geometry.honeycomb_lattice()
    scf = VJinteraction(g.get_hamiltonian(), U=3.0, V1=0.5, filling=0.5,
            mf="antiferro", nk=nk, mix=0.5, maxerror=1e-10, maxite=600)
    assert scf.converged
    assert scf.residual < 1e-10
    assert scf.iterations < 400
    assert abs(scf.hamiltonian.get_goldstone_residual(nk=nk)) < 1e-8
