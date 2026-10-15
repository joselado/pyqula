import numpy as np

from pyqula import geometry
from pyqula.transporttk.localprobe import LocalProbe
from pyqula.keldyshtk.current import build_shared_selfenergy, dc_current

# The STM-tip conductance of the user guide (documentation/figures/
# transport_stmtip.py) returned one point, eV=0.2725, at half its value
# with no warning when run with tol=2e-3, nmax_max=40. The sidebands were
# not the cause: the quasienergy quadrature accepted the panel holding the
# gap-edge peak at eV=Delta on a Gauss-Kronrod estimate that agreed with
# itself by coincidence, so the current at voltage+dv came out 1.2% low,
# and the finite difference of keldysh_didv amplifies that by
# Ip/(Ip-Im), about 40 here. The coincidence depends on the exact
# integrand, which here comes from the shared AAA fit of the self-energies
# that didv(energies=...) builds for the whole sweep (vmax=0.4, nmax_max=40),
# so a change of that fit can make this case stop discriminating; the
# synthetic test_one_coincidental_estimate_does_not_end_the_refinement in
# test_adaptive_quad_batch.py pins the mechanism independently of it.


def _stm_tip():
    g = geometry.chain()
    h = g.get_hamiltonian(); h.shift_fermi(1.); h.add_swave(0.1)
    lead = geometry.chain().get_hamiltonian(); lead.shift_fermi(1.)
    lead.add_swave(0.1)
    lp = LocalProbe(h, lead=lead, delta=1e-3)
    lp.T = 0.3
    return lp


def test_localprobe_quadrature_gap_edge():
    """The adaptive quadrature must agree with the deterministic composite
    Gauss-Legendre rule at the voltage that failed, and the loose setting
    (tol=2e-3, nmax_max=40, with the shared fit of a sweep) must agree with
    the tight one (tol=5e-4, nmax_max=60, direct self-energies) for the
    dI/dV. The tolerance on dI/dV is 5e-3 rather than 1e-3, since the
    finite difference amplifies the 1e-3 of the quadrature; before the fix
    the two were a factor 1.85 apart."""
    lp = _stm_tip()
    shared = build_shared_selfenergy(lp, 0.4, nmax_max=40, delta=1e-3)
    assert shared is not None
    V = 0.2725*1.01 # the voltage+dv of keldysh_didv's finite difference
    kw = dict(delta=1e-3, fixed_nmax=32, selfenergy_qtci=shared)
    Iad = dc_current(lp, V, **kw)
    Ifix = dc_current(lp, V, quadrature="fixed", **kw)
    assert abs(Iad-Ifix) <= 1e-3*abs(Ifix)
    Gloose = lp.didv(energy=0.2725, nmax=4, nmax_max=40, tol=2e-3,
                     selfenergy_qtci=shared)
    Gtight = lp.didv(energy=0.2725, nmax=4, nmax_max=60, tol=5e-4)
    assert abs(Gloose-Gtight) <= 5e-3*abs(Gtight)
