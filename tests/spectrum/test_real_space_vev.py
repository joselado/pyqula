import numpy as np

from pyqula import geometry, spectrum


def _chain(n, has_spin=True):
    g = geometry.chain().supercell(n)
    g.dimensionality = 0
    return g.get_hamiltonian(has_spin=has_spin)


def test_imaginary_operator_keeps_its_sign():
    """densitymatrix.full_dm builds the transpose of rho, so contracting
    it untransposed evaluates <A*> instead of <A> -- a sign flip for any
    purely imaginary operator (sy, the valley operator, a current). The
    same defect fbee7c9 fixed in spectrum.ev, in a sibling three lines
    below it."""
    h = _chain(4)
    h.add_exchange([0.3, 0.5, 0.2])
    h.add_onsite(0.2)
    (es, ws) = h.get_eigenvectors()
    for name in ["sx", "sy", "sz"]:
        op = h.get_operator(name)
        m = op.get_matrix().toarray()
        ref = sum([np.conjugate(w).dot(m @ w) for (e, w) in zip(es, ws)
                   if e < 0.]).real
        out = np.sum(spectrum.real_space_vev(h, operator=op, nk=1))
        assert abs(out - ref) < 1e-8, name


def test_nambu_agrees_with_the_normal_state():
    """With the electron-hole degree of freedom the sum runs over the
    particle-hole-redundant BdG states and full2profile then adds the
    electron and hole entries of each site, so every site came out as
    exactly 2.0 whatever the density was. A BdG description of a state
    must return the same profile as the normal-state description of it,
    the convention get_vev uses."""
    h = _chain(3)
    h.add_onsite(0.3)
    hn = h.copy()
    hn.setup_nambu_spinor()
    hn.add_swave(0.0)  # no pairing: the same physical state
    normal = spectrum.real_space_vev(h, nk=1)
    nambu = spectrum.real_space_vev(hn, nk=1)
    assert np.max(np.abs(normal - nambu)) < 1e-8
    assert np.max(np.abs(normal - h.get_vev())) < 1e-8
    # and the profile must not be a featureless constant
    assert np.max(normal) - np.min(normal) > 1e-3


def test_default_operator_is_the_density():
    """operator=None is the documented default but died in
    operators.Operator(None) on a bare `raise`."""
    h = _chain(3)
    h.add_onsite(0.3)
    assert np.max(np.abs(spectrum.real_space_vev(h, nk=1)
                         - h.get_vev())) < 1e-8


def test_nrep_is_not_ignored(tmp_path, monkeypatch):
    """The signature declares nrep=3; the body hardcoded nrep=5 in the
    write_profile call."""
    monkeypatch.chdir(tmp_path)
    g = geometry.chain().supercell(3)  # periodic: nrep actually replicates
    h = g.get_hamiltonian()
    h.add_onsite(0.3)
    counts = []
    for nrep in [1, 4]:
        spectrum.real_space_vev(h, nk=1, nrep=nrep, name="PROF.OUT")
        counts.append(len(np.genfromtxt("PROF.OUT")))
    assert counts[0] != counts[1]
