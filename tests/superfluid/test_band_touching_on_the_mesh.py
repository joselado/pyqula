"""The conventional/geometric split with a band touching on the k-mesh.

The kagome lattice has a Dirac point at K = (1/3,2/3), which lies on the
nk x nk mesh whenever nk is a multiple of 3. The decomposition used to
raise there ("degenerate normal-state bands with a finite interband
current"), while nk=20, 22, 31, 40, 41 all gave the same converged split:
the split is a Brillouin-zone integral of a bounded integrand, and only
the one mesh point sitting on the touching is ambiguous, because the
band basis there depends on the direction from which it is approached.
That point now keeps its (basis independent) total and takes its
conventional part as the average over its neighbours."""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.sctk import superfluidweight as sw


def _kagome(mu=1.0, delta=0.2):
    """Kagome lattice with the chemical potential at the Dirac point
    (E=1 for the flat band at E=-2), where the touching weighs most."""
    h = geometry.kagome_lattice().get_hamiltonian()
    h.add_onsite(-mu)
    h.add_swave(delta)
    return h


def test_kagome_dirac_point_on_the_mesh():
    """nk=21 puts K on the mesh. The split must exist, add up to the Kubo
    weight exactly, keep the C3 isotropy of the lattice and agree with a
    mesh that misses K. Taking the split at K in whatever basis eigh
    returns instead gives a conventional part of 0.0768, a third too large
    against the 0.0577 of nk=40, which is what the last check excludes."""
    h = _kagome()
    out = sw.superfluid_weight_decomposition(h, nk=21, T=0.)
    tot = sw.superfluid_weight(h, nk=21, T=0.)
    assert np.max(np.abs(out["total"]-tot))/np.max(np.abs(tot)) < 1e-10
    c = out["conventional"]
    assert abs(c[0, 0]-c[1, 1]) < 1e-4*c[0, 0], c # C3: xx = yy
    assert abs(c[0, 1]) < 1e-4*c[0, 0], c # and no xy
    ref = sw.superfluid_weight_decomposition(h, nk=40, T=0.)["conventional"]
    assert abs(c[0, 0]-ref[0, 0]) < 3e-2*ref[0, 0], (c, ref) # 0.0572 vs 0.0577


def test_kramers_pairs_with_inversion_are_not_touchings():
    """With inversion and time-reversal symmetry every band is a Kramers
    pair, and every current operator is a multiple of the identity on the
    pair, so no k-point is a touching in the sense above and the split
    goes through directly."""
    h = geometry.buckled_honeycomb_lattice().get_hamiltonian()
    h.add_onsite(0.3)
    h.add_soc(0.1)
    h.add_swave(0.2)
    out = sw.superfluid_weight_decomposition(h, nk=9, T=0.)
    tot = sw.superfluid_weight(h, nk=9, T=0.)
    assert np.max(np.abs(out["total"]-tot))/np.max(np.abs(tot)) < 1e-10
    assert out["conventional"][0, 0] > 0. and out["geometric"][0, 0] > 0.


def test_line_of_touchings_along_an_axis_raises():
    """Two identical square-lattice layers coupled by lam sin(k_y) tau_y
    are degenerate on the whole line k_y=0, with a finite current J_y
    between them. Moving along x stays on the line, so the average over
    the neighbours of a mesh point cannot resolve it, and that has to
    raise rather than return a split sampled in an arbitrary basis."""
    g = geometry.square_lattice()
    g.r = np.array([[0., 0., 0.], [0., 0., 1.]]) # two layers
    g.r2xyz()
    lam = 0.4
    def f(r1, r2):
        dr = r2-r1
        if abs(dr[2]) < 1e-6 and abs(np.hypot(dr[0], dr[1])-1.) < 1e-6:
            return -1.0 # first neighbors inside a layer
        if (abs(abs(dr[2])-1.) < 1e-6 and abs(dr[0]) < 1e-6
                and abs(abs(dr[1])-1.) < 1e-6):
            return -lam/2.*dr[1]*dr[2] # lam sin(k_y) tau_y
        return 0.
    h = g.get_hamiltonian(fun=f)
    h.add_onsite(0.7)
    h.add_swave(0.3)
    with pytest.raises(ValueError, match="line of band touchings"):
        sw.superfluid_weight_decomposition(h, nk=6, T=0.)
    assert sw.superfluid_weight(h, nk=6, T=0.)[0, 0] > 0. # Kubo still works
