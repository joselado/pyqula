import numpy as np
import pytest

from pyqula import geometry, dos

# dos.surface_dos declared an `operator` argument and never consumed it,
# so surface_dos(h,operator="sz") returned the plain charge surface DOS,
# byte for byte. dos.get_dos in the same module honours the identical
# argument, and kdos's surface routines were repaired for this in a39bf68
# while this copy was not.
#
# The oracle is not a recorded curve: a Zeeman-polarized Hamiltonian is
# block diagonal in spin, so its surface Green's function is too, and the
# sz-projected surface DOS must be exactly the difference of the two
# spinless surface DOS's computed with the two shifted onsite energies
# (and the unprojected one exactly their sum). Both come out of this same
# function, so nothing outside it is assumed.

BZ = 0.4


def _spin_blocks(g):
    """The two spin blocks of a Zeeman-polarized Hamiltonian on g, as
    independent spinless Hamiltonians."""
    hup = g.get_hamiltonian(has_spin=False)
    hup.add_onsite(BZ)
    hdn = g.get_hamiltonian(has_spin=False)
    hdn.add_onsite(-BZ)
    return (hup, hdn)


def _polarized(g):
    h = g.get_hamiltonian()
    h.add_zeeman([0., 0., BZ])
    return h


def test_surface_dos_operator_resolves_into_the_spin_blocks_1d():
    g = geometry.chain()
    energies = np.linspace(-.6, .6, 13)
    kw = dict(energies=energies, delta=0.05)
    total = dos.surface_dos(_polarized(g), **kw)[1]
    sz = dos.surface_dos(_polarized(g), operator="sz", **kw)[1]
    (hup, hdn) = _spin_blocks(g)
    up = dos.surface_dos(hup, **kw)[1]
    dn = dos.surface_dos(hdn, **kw)[1]
    assert np.max(np.abs(total - (up + dn))) < 1e-10
    assert np.max(np.abs(sz - (up - dn))) < 1e-10
    # and the projection is not a no-op: it used to be
    assert np.max(np.abs(total - sz)) > 1e-2


def test_surface_dos_operator_resolves_into_the_spin_blocks_2d():
    """The 2d branch averages the surface Green's function over a chain
    of kpoints, a separate call site from the 1d one. The tolerance here
    is the renormalization solver's own, which stops at a finite residual
    per kpoint rather than at machine precision."""
    g = geometry.square_lattice()
    energies = np.linspace(-.6, .6, 5)
    kw = dict(energies=energies, delta=0.1,
              klist=[[k, 0., 0.] for k in np.linspace(-.5, .5, 6)])
    total = dos.surface_dos(_polarized(g), **kw)[1]
    sz = dos.surface_dos(_polarized(g), operator="sz", **kw)[1]
    (hup, hdn) = _spin_blocks(g)
    up = dos.surface_dos(hup, **kw)[1]
    dn = dos.surface_dos(hdn, **kw)[1]
    assert np.max(np.abs(total - (up + dn))) < 1e-7
    assert np.max(np.abs(sz - (up - dn))) < 1e-7
    assert np.max(np.abs(total - sz)) > 1e-2


def test_surface_dos_refuses_a_momentum_dependent_operator():
    """The Green's function is already summed over the transverse
    kpoints when the operator would be applied, so a k-dependent one
    cannot be honoured -- better to say so than to drop it silently,
    which is what the whole argument used to do."""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    with pytest.raises(NotImplementedError):
        dos.surface_dos(h, energies=np.linspace(-.2, .2, 3),
                        operator="valley")
