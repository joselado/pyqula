"""fermi_surface_generator declared refine_delta and never referenced it,
leaving the orphan comment 'setup a reasonable value for delta' with
nothing after it. The argument refines the broadening -- that is what it
did before the automatic-delta line it belonged to was dropped
(fermisurfacetk/singlefs.py still carries that line) -- so
refine_delta=r must be the same calculation as delta/r, and only that."""

import numpy as np

from pyqula import geometry
from pyqula.fermisurface import fermi_surface_generator


def _h():
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_zeeman([0., 0., 0.3])
    return h


def test_refine_delta_divides_the_broadening():
    """The invariant: refining by r is exactly the same Fermi surface as
    asking for a broadening r times narrower. Before, refine_delta=50 and
    refine_delta=1 gave identical weights."""
    h = _h()
    _, _, coarse = fermi_surface_generator(h, energies=[0.2], nk=5, delta=0.1)
    _, _, fine = fermi_surface_generator(h, energies=[0.2], nk=5, delta=0.1,
                                            refine_delta=5.)
    _, _, ref = fermi_surface_generator(h, energies=[0.2], nk=5, delta=0.02)
    assert np.allclose(fine, ref)          # refine_delta=r is delta/r
    assert not np.allclose(fine, coarse)   # and it really does something


def test_refine_delta_one_is_the_untouched_broadening():
    """The default must change nothing: refine_delta=1.0 is the delta the
    caller passed, bit for bit."""
    h = _h()
    _, _, a = fermi_surface_generator(h, energies=[0.0, 0.2], nk=4, delta=0.1)
    _, _, b = fermi_surface_generator(h, energies=[0.0, 0.2], nk=4, delta=0.1,
                                        refine_delta=1.0)
    assert np.array_equal(a, b)
