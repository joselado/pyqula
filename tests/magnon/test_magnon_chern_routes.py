"""The Chern numbers of the magnons of one saturated ferromagnet from the
three routes of the package, on the same mean field, the same k-mesh and
the same windows of frequency:

- route 1, bsetk/topology.py: the eigenvectors of time-dependent
  Hartree-Fock lifted to the orbital pair basis, and the plaquette product
  on a mesh of Q;
- route 2, chitk/kernelchern.py: the null vectors of the kernel
  K = chi0^-1 - V of the site-basis spin RPA in the raising channel, at its
  zeros, and the same plaquette product;
- route 3, chitk/responsewinding.py: the winding of the same kernel in the
  raising channel over a contour around a window, with no eigenvector at
  all.

The magnet has an onsite U only, so the site-basis RPA and the TDHF
describe the same modes, and a Zeeman field gaps its Goldstone mode. Its
two magnon bands are at 4.55 to 4.79 and 5.15 to 5.91 on the mesh used here
(nk=6, nq=6), and the Stoner continuum is 1.89 or more above the upper one,
so each of the windows (4.0, 4.964) and (4.964, 6.687) holds one band.

The integers alone are a weak comparison. The null vectors of routes 2
and 3 live in the two-dimensional space of S^+ on the two sites, so their
upper band has minus the Chern number of the lower one and the pair has 0
for any vectors; only the lower band and its sign under the Kane-Mele
coupling are measurements there. Route 1 lives in the pair basis, where
the upper band and the pair are measurements too. So the modules are first
linked beyond the integers: at the same Q the zeros of route 2 are the
energies of route 1 and its null vector is the site part of the state of
route 1, and at -Q the kernel of route 3 is singular at the energy of
route 1, with the complex conjugate of that site part as its null vector.
The conjugation is why route 3 returns minus the winding of its kernel.
"""
import functools

import numpy as np
import pytest

from pyqula import geometry
from pyqula.bsetk import topology as bt
from pyqula.bsetk.spinflip import magnon_energies
from pyqula.chitk import kernelchern as kc
from pyqula.chitk import responsewinding as rw

NK = 6 # the k-mesh of the mean field and of the three routes
NQ = 6 # the mesh of Q of the plaquette product
SHIFT = (0.03, 0.07) # keeps Q=0 and the corners out of the mesh of Q
T = 1e-3 # the temperature of the occupations of the two bubbles
LOWER, UPPER, BOTH = (4.0, 4.964), (4.964, 6.687), (4.0, 6.687)


@functools.lru_cache(maxsize=None)
def _magnet(lam):
    """The honeycomb ferromagnet saturated along -z by its field"""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(tij=[1.0, 0.3])
    h.add_kane_mele(lam)
    h.add_zeeman([0., 0., 3.0])
    hmf = h.get_mean_field_hamiltonian(U=8.0, filling=0.5, mf="ferro",
            nk=NK, maxerror=1e-10)
    assert np.allclose(np.real(hmf.get_vev("sz")), -1., atol=1e-8)
    return hmf


def _overlap(a, b):
    return abs(np.vdot(a, b))/np.linalg.norm(a)/np.linalg.norm(b)


def _site_part(h, v):
    """The amplitude of a lifted magnon v on S^+_i = c^dag_{i,up} c_{i,dn}:
    the element (i up, i dn) of its resonant block Phi(k;a,b), summed over
    k. A saturated magnet has no de-excitation half, so the antiresonant
    block of v is zero"""
    n = 2*len(h.geometry.r)
    assert np.linalg.norm(v[len(v)//2:]) == 0.
    Phi = v[0:len(v)//2].reshape(-1, n, n).sum(axis=0)
    return np.array([Phi[2*i, 2*i+1] for i in range(n//2)])


def test_route_2_has_the_modes_and_the_states_of_route_1():
    """At every point of the mesh of route 1 the zeros of the kernel of
    route 2 are the energies of its record (5e-14 measured), since both
    take the pairs of an electron at k+Q and a hole at k. At two generic
    Q the null vector of route 2 is the site part of the lifted state of
    route 1 (overlap 1 to 1e-15), and not its complex conjugate (0.943 and
    0.989 measured), which would give the opposite integer"""
    h = _magnet(0.15)
    r = bt.magnon_chern(h, bands=[0, 1], nq=NQ, shift=SHIFT, nk=NK,
            full_output=True)
    K = kc.spin_rpa_kernel(h, nk=NK, T=T)
    for q, e in zip(r.qs.reshape(-1, 3), r.energies.reshape(-1, 2)):
        assert np.max(np.abs(kc.kernel_modes(K, q, BOTH)[0] - e)) < 1e-10
    for Q in ([0.21, 0.37, 0.], [0.13, 0.29, 0.]):
        es, vs = kc.kernel_modes(K, Q, BOTH)
        for band in (0, 1):
            phi = _site_part(h, bt.mode_states(h, "magnon", Q=Q, bands=[band],
                    nk=NK)[0])
            assert _overlap(vs[band], phi) > 1. - 1e-10
            assert _overlap(vs[band], np.conj(phi)) < 1. - 1e-3


def test_route_3_at_minus_q_has_the_mode_of_route_1_at_q():
    """On a Q of the k-mesh the bubble of route 3 at -Q and the pair basis
    of route 1 at Q sum over the same pairs (off the mesh they differ at
    finite nk). There the response kernel of the raising channel, S^+ on
    the two sites, is singular at the energy of route 1, and its null
    vector is conj(phi), with phi the site part of the state of route 1 on
    S^+, and not phi itself"""
    h = _magnet(0.15)
    ginv, hermitian = rw.spin_kernel(h, nk=NK, T=T, channel="raising",
                                     kernel="response")
    assert hermitian
    for Q in (np.array([1./NK, 2./NK, 0.]), np.array([2./NK, 1./NK, 0.])):
        e = np.sort(magnon_energies(h, Q=Q, nk=NK).real)[0:2]
        for band in (0, 1):
            phi = _site_part(h, bt.mode_states(h, "magnon", Q=Q, bands=[band],
                    nk=NK)[0])
            m = ginv(np.array([e[band] + 0j]), -Q)[0]
            ev, u = np.linalg.eigh((m + m.conj().T)/2.)
            i = int(np.argmin(np.abs(ev)))
            assert abs(ev[i]) < 1e-10*np.max(np.abs(ev))
            assert _overlap(u[:, i], np.conj(phi)) > 1. - 1e-10
            assert _overlap(u[:, i], phi) < 1. - 1e-5


def test_routes_1_and_2_give_the_same_chern_numbers():
    """The lower band, the upper band and the pair, for both signs of the
    Kane-Mele coupling, whose product with t' plays the role of a
    Dzyaloshinskii-Moriya term. Each window holds its group and nothing
    else at every Q of the mesh, which is read from the record of route 1.
    Measured: +1, -1 and 0, and -1, +1 and 0 with the opposite coupling, the
    same at (nk, nq) = (8, 8) and (8, 10)"""
    out = dict()
    for lam in (0.15, -0.15):
        h = _magnet(lam)
        r = bt.magnon_chern(h, bands=[0, 1], nq=NQ, shift=SHIFT, nk=NK,
                full_output=True)
        e0, e1 = r.energies[:, :, 0], r.energies[:, :, 1]
        assert LOWER[0] < np.min(e0) and np.max(e0) < LOWER[1] < np.min(e1)
        # the continuum is at least r.bound above the upper band at every Q
        assert np.max(e1) < UPPER[1] < np.min(e1) + r.bound
        K = kc.spin_rpa_kernel(h, nk=NK, T=T)
        for bands, window in (([0], LOWER), ([1], UPPER), ([0, 1], BOTH)):
            r = bt.magnon_chern(h, bands=bands, nq=NQ, shift=SHIFT, nk=NK,
                    full_output=True)
            assert r.gap > 0.1 and r.bound > 1. # 0.36 and 1.89 measured
            c1 = r.chern
            c2 = kc.kernel_chern(K, window, nq=NQ, shift=SHIFT)
            assert abs(c1 - round(c1)) < 1e-8 and abs(c2 - c1) < 1e-8
            out[lam, len(bands), bands[0]] = round(c1)
    for lam in (0.15, -0.15):
        assert abs(out[lam, 1, 0]) == 1 # a Chern band, not 0 against 0
        assert out[lam, 1, 1] == -out[lam, 1, 0] and out[lam, 2, 0] == 0
    assert out[-0.15, 1, 0] == -out[0.15, 1, 0]


@pytest.mark.slow
def test_route_3_winds_to_the_integers_of_route_1():
    """The winding is not an integer on a finite mesh. At nq=8, with the
    shift of the other routes, the lower window gives 0.9950 at nz=4, and
    1.029 and 1.002 on the neighbouring meshes nq=7 and 9. The upper window
    is wider, so its contour is taller against the distance of its ends to
    the bands, and needs nz=8: -0.9953, against -1.025 at nz=4 (-1.0002
    at nq=12). The opposite coupling gives -0.9950. The tolerance is four
    times the error at this mesh; a wrong sign or a missing factor is off
    by one or more"""
    kw = dict(nq=8, nk=NK, T=T, channel="raising", shift=SHIFT)
    signs = []
    for lam in (0.15, -0.15):
        h = _magnet(lam)
        cases = (([0], LOWER, 4), ([1], UPPER, 8)) if lam > 0. else \
                (([0], LOWER, 4),)
        for bands, window, nz in cases:
            c1 = round(bt.magnon_chern(h, bands=bands, nq=NQ, shift=SHIFT,
                    nk=NK))
            assert abs(c1) == 1
            c3 = rw.spin_response_chern(h, window, nz=nz, **kw)
            assert abs(c3 - c1) < 2e-2
            if bands == [0]: signs.append(c1)
    assert signs[1] == -signs[0]


def test_route_3_selects_the_sector_of_a_ferrimagnet():
    """The Lieb ferrimagnet has magnon bands in both spin sectors, which
    overlap in energy: its two bands of route 1 are in the raising channel
    and the band of the other sector, in the lowering one, crosses their
    gap. Route 3 in the raising channel, with windows read from the record
    of route 1, gives their +1 and -1 (1.0048 and -0.9985 measured at
    nq=8, nz=4), as route 2 in the same channel does"""
    g = geometry.lieb_lattice()
    h = g.get_hamiltonian(tij=[1.0, 0.3])
    h.add_kane_mele(0.15)
    h.add_zeeman([0., 0., 0.3])
    h = h.get_mean_field_hamiltonian(U=4.0, filling=0.5, mf="ferro", nk=NK,
            maxerror=1e-10, mix=0.3, maxite=3000)
    r0, r1 = [bt.magnon_chern(h, bands=[b], nq=NQ, shift=SHIFT, nk=NK,
              full_output=True) for b in (0, 1)]
    c1 = [round(r0.chern), round(r1.chern)]
    assert abs(c1[0]) == 1 and c1[1] == -c1[0]
    e0, e1 = r0.energies[:, :, 0], r1.energies[:, :, 0]
    mid = 0.5*(e0.max() + e1.min())
    windows = ((0.5*e0.min(), mid), (mid, e1.max() + 0.5*r1.bound))
    K = kc.spin_rpa_kernel(h, nk=NK, channel="raising")
    for w, c in zip(windows, c1):
        assert abs(kc.kernel_chern(K, w, nq=NQ, shift=SHIFT) - c) < 1e-8
        c3 = rw.spin_response_chern(h, w, channel="raising", nq=8, nz=4,
                                    nk=NK, shift=SHIFT)
        assert abs(c3 - c) < 2e-2
