"""specialhamiltonian.twisted_bilayer_graphene at the smallest commensurate
moire index (n=1).

On the constant this file used to pin, and why there is no longer one.
Two values have been committed here in turn -- -13.738703648103538
(`6c8cfbc`) and -13.735331753001446 (`efd3cf3`) -- each as "the"
reproducible band sum, each time with the other declared unreproducible.
Both are right, and the quantity is the problem.

`np.sum(e)` there was the sum of the eight eigenvalues nearest zero at each
of 13 k-points, taken after `h.set_filling(0.5, nk=1)`. Three things make
that a measurement of the Fermi offset rather than of the Hamiltonian:

  - `set_filling(0.5, nk=1)` estimates the Fermi energy from a single
    Gamma point, `(E[13]+E[14])/2` of 28 levels. It is nowhere near
    converged -- over nk = 1,2,3,4,6,8,12 it returns 0.1439, 0.0985,
    0.0035, 0.0489, 0.0035, 0.0175, 0.0035. Whatever any part of that
    estimate depends on, the offset follows.
  - all 104 eigenvalues carry that offset, so the sum amplifies it 104x.
    The two committed values differ by 0.0033719, which is exactly a
    3.24e-5 change in the offset; `atol=1e-6` on the sum was pinning the
    Fermi energy to 1e-8.
  - "the eight nearest zero" is a discontinuous selection on top. Move the
    offset by 1e-3 and the sum jumps from -13.74 to -10.17.

So the pin was never discriminating the Hamiltonian; it was asserting that
`set_filling`'s single-k-point estimate had not moved in its eighth
decimal. What this file pins instead is shift-invariant by construction --
quantities that cannot move when an onsite term does, asserted to be
invariant rather than assumed to be -- plus the exact structure of the
moire cell. The bilayer character is checked separately at the end, by the
degeneracy the interlayer hopping lifts.
"""
import numpy as np
import scipy.linalg as lg

from pyqula import algebra, klist, specialhamiltonian
from pyqula.kpointstk.labels import label2k


KPATH = ["G", "K", "M", "K'", "G"]


def _tbg(ti=0.4):
    return specialhamiltonian.twisted_bilayer_graphene(n=1, ti=ti,
                                                        has_spin=False)


def _bloch(h, k):
    return np.array(algebra.todense(h.get_hk_gen()(k)), dtype=np.complex128)


def _centred(m):
    """`m` with its mean diagonal removed, so that every trace taken of it
    below is unchanged by any uniform onsite shift -- `set_filling`'s
    included."""
    n = m.shape[0]
    return m - (np.trace(m)/n)*np.identity(n)


def _fingerprint(m):
    """Three numbers per k-point, none of which an onsite shift can move:
    the second and third moments of the centred Bloch matrix (pure
    matrix algebra, no eigensolver at all, so they carry no LAPACK
    dependence) and the total bandwidth (a difference of eigenvalues)."""
    c = _centred(m)
    ev = lg.eigvalsh(m)
    return (np.trace(c@c).real, np.trace(c@c@c).real, ev.max()-ev.min())


# Fingerprints of the n=1, ti=0.4 Hamiltonian at the three high-symmetry
# points, as (Tr C^2, Tr C^3, bandwidth). Recorded to 12 significant
# digits; the first two are reproducible to ~1e-15 relative (matmul and
# trace only) and the third to ~1e-13 (one dense eigenvalue solve).
FINGERPRINT = {
    "G": (87.8283716074, -17.3913013770, 6.70043276976),
    "K": (87.8283053575, -17.3764033486, 5.49735492095),
    "M": (87.8283130643, -17.3780567879, 5.76237957485),
}


def test_twisted_bilayer_graphene_bands_match_dense_diagonalization(tmp_path,
                                                                    monkeypatch):
    """The sparse (ARPACK) band energies along G-K-M-K'-G must be the eight
    eigenvalues closest to zero of the dense Bloch matrix at the same
    k-points.

    `set_filling` is kept here because it is what puts the eight bands of
    interest around zero for ARPACK to find, but it cancels between the two
    sides -- both are computed from the same shifted `h` -- so nothing
    asserted below depends on the value it picks.
    """
    monkeypatch.chdir(tmp_path)  # get_bands writes BANDS.OUT to cwd
    h = _tbg()
    h.set_filling(0.5, nk=1)
    (k, e) = h.get_bands(num_bands=8, kpath=KPATH, nk=8)
    assert e.shape == (104,)
    ks = klist.get_kpath(h.geometry, kpath=KPATH, nk=8)
    assert len(ks)*8 == len(e)
    ref = []
    margin = np.inf
    for ki in ks:
        ev = lg.eigvalsh(_bloch(h, ki))
        order = np.argsort(np.abs(ev))
        ref += sorted(ev[order[:8]])  # the eight nearest zero
        # "nearest zero" is a discontinuous choice, so record how far the
        # two solvers are from disagreeing about it rather than finding
        # out by flapping
        a = np.sort(np.abs(ev))
        margin = min(margin, a[8]-a[7])
    assert np.allclose(e, ref, atol=1e-10), np.max(np.abs(e-np.array(ref)))
    assert margin > 1e-5, ("the 8th and 9th levels are nearly equidistant "
                           "from zero (margin %g): which eight bands this "
                           "test compares is no longer well defined" % margin)


def test_twisted_bilayer_graphene_cell_is_the_n1_commensurate_one(tmp_path,
                                                                  monkeypatch):
    """The n=1 commensurate cell is the sqrt(7)xsqrt(7) supercell of
    graphene at 21.787 degrees: 28 sites, and lattice vectors of length
    sqrt(7)*sqrt(3) = sqrt(21) at 60 degrees. These are exact algebraic
    values, not recorded measurements."""
    monkeypatch.chdir(tmp_path)
    h = _tbg()
    assert len(h.geometry.r) == 28
    assert h.intra.shape[0] == 28
    assert h.dimensionality == 2
    a1, a2 = np.array(h.geometry.a1), np.array(h.geometry.a2)
    assert np.isclose(np.linalg.norm(a1), np.sqrt(21), rtol=0, atol=1e-12)
    assert np.isclose(np.linalg.norm(a2), np.sqrt(21), rtol=0, atol=1e-12)
    cos = np.dot(a1, a2)/(np.linalg.norm(a1)*np.linalg.norm(a2))
    assert np.isclose(cos, 0.5, rtol=0, atol=1e-12)  # 60 degrees


def test_twisted_bilayer_graphene_fingerprint_is_the_recorded_hamiltonian(
        tmp_path, monkeypatch):
    """The Hamiltonian itself, through quantities an onsite shift cannot
    move -- which is what the old band sum failed to be."""
    monkeypatch.chdir(tmp_path)
    h = _tbg()
    for label, expected in FINGERPRINT.items():
        got = _fingerprint(_bloch(h, label2k(h.geometry, label)))
        assert np.allclose(got[:2], expected[:2], rtol=1e-10, atol=0), \
            (label, got, expected)
        assert np.isclose(got[2], expected[2], rtol=1e-9, atol=0), \
            (label, got, expected)


def test_twisted_bilayer_graphene_fingerprint_survives_any_onsite_shift(
        tmp_path, monkeypatch):
    """The property the old pin lacked, asserted rather than assumed: an
    arbitrary shift of the Fermi level -- `set_filling`'s, or any other --
    leaves every fingerprint above where it was. A future change to how the
    filling is estimated therefore cannot make this file red.

    `Tr C^2` comes back bit-identical, since removing the mean diagonal is
    exact for it. `Tr C^3` and the bandwidth pick up ordinary rounding from
    the extra matmul and the eigenvalue solve -- ~2e-15 relative, thirteen
    orders of magnitude under the 3.24e-5 offset change that split the two
    historical pins."""
    monkeypatch.chdir(tmp_path)
    ref = _tbg()
    shifted = [_tbg() for _ in range(3)]
    for h, shift in zip(shifted, (0.7137, -2.5, 1e-8)):
        h.shift_fermi(shift)
    # the filling call the first test makes is just such a shift, so it
    # belongs in the same list rather than in a check of its own
    filled = _tbg()
    filled.set_filling(0.5, nk=1)
    for h in shifted+[filled]:
        for label in FINGERPRINT:
            k = label2k(h.geometry, label)
            a = _fingerprint(_bloch(h, k))
            b = _fingerprint(_bloch(ref, k))
            assert a[0] == b[0], (label, a, b)  # exact
            assert np.allclose(a[1:], b[1:], rtol=1e-12, atol=0), (label, a, b)


def test_twisted_bilayer_graphene_fingerprint_moves_with_the_interlayer_hopping(
        tmp_path, monkeypatch):
    """A fingerprint nothing can move would pin nothing. `Tr C^2` is the
    sum of the squared hopping amplitudes, so switching the interlayer
    coupling off has to drop it to the two decoupled monolayers' value --
    84, i.e. 28 sites x 3 neighbours x 1^2 -- and `Tr C^3`, a sum over
    closed triangles, essentially to zero."""
    monkeypatch.chdir(tmp_path)
    k = label2k(_tbg().geometry, "G")
    c2_on, c3_on, _ = _fingerprint(_bloch(_tbg(ti=0.4), k))
    c2_off, c3_off, _ = _fingerprint(_bloch(_tbg(ti=0.0), k))
    assert np.isclose(c2_off, 84.0, rtol=0, atol=1e-4), c2_off
    assert abs(c3_off) < 0.1, c3_off
    assert c2_on-c2_off > 3.5, (c2_on, c2_off)
    assert abs(c3_on) > 15., c3_on


def _max_layer_pair_splitting(ti, kpath, nk=8):
    """Largest splitting of consecutive level pairs along the k-path.

    With the two layers decoupled every eigenvalue of the bilayer is a
    doubled monolayer eigenvalue, so sorting the spectrum and differencing
    it in pairs gives zero; the interlayer hopping is what lifts it."""
    h = _tbg(ti=ti)
    h.set_filling(0.5, nk=1)
    hk = h.get_hk_gen()
    out = 0.
    for ki in klist.get_kpath(h.geometry, kpath=kpath, nk=nk):
        ev = np.sort(lg.eigvalsh(np.array(algebra.todense(hk(ki)),
                                          dtype=np.complex128)))
        out = max(out, np.max(np.abs(ev[1::2]-ev[0::2])))
    return out


def test_twisted_bilayer_interlayer_hopping_splits_the_layer_degeneracy(
        tmp_path, monkeypatch):
    """The fingerprints above cannot tell a bilayer from two decoupled
    monolayers stacked without coupling, which is the one thing
    `twisted_bilayer_graphene` exists to build. At ti=0 the layers decouple
    and the spectrum is exactly the monolayer one doubled, so consecutive
    levels are degenerate; ti=0.4 splits them by an amount of order the
    hopping itself."""
    monkeypatch.chdir(tmp_path)
    assert _max_layer_pair_splitting(0.0, KPATH) < 1e-10
    assert _max_layer_pair_splitting(0.4, KPATH) > 0.5
