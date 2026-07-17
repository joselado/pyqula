import os

import numpy as np
import scipy.sparse.linalg as slg

from pyqula import geometry
from pyqula import operators
from pyqula import bandstructure
from pyqula.algebra import braket_wAw
from pyqula import algebra


def _legacy_get_bands_nd(h, kpath=None, operator=None, num_bands=None,
                          callback=None, central_energy=0.0, nk=400,
                          ewindow=None, output_file="BANDS_LEGACY.OUT",
                          write=True, silent=True):
    """Verbatim copy of the pre-refactor get_bands_nd, which stored results
    as a formatted string and parsed them back into an array at the end.
    Kept here only so the new numeric implementation can be checked against
    it; not part of the library."""
    if num_bands is not None:
        if num_bands > (h.intra.shape[0] - 1):
            num_bands = None
    if operator is not None:
        operator = h.get_operator(operator)
    if num_bands is None:
        if operator is not None:
            def diagf(m):
                return algebra.eigh(m)
        else:
            def diagf(m):
                return algebra.eigvalsh(m)
    else:
        h = h.copy()
        h.turn_sparse()

        def diagf(m):
            eig, eigvec = slg.eigsh(m, k=num_bands, which="LM",
                                     sigma=central_energy,
                                     tol=bandstructure.arpack_tol,
                                     maxiter=bandstructure.arpack_maxiter)
            if operator is None:
                return eig
            else:
                return (eig, eigvec)
    hkgen = h.get_hk_gen()
    kpath = h.geometry.get_kpath(kpath, nk=nk)

    def getek(k):
        out = ""
        hk = hkgen(kpath[k])
        if operator is None:
            es = diagf(hk)
            es = np.sort(es)
            for e in es:
                out += str(k) + "   " + str(e) + "\n"
            if callback is not None:
                callback(k, es)
        else:
            es, ws = diagf(hk)
            ws = ws.transpose()

            def evaluate(w, k, A):
                if type(A) == operators.Operator:
                    waw = A.braket(w, k=kpath[k]).real
                elif callable(A):
                    try:
                        waw = A(w, k=kpath[k])
                    except Exception:
                        waw = A(w)
                else:
                    waw = braket_wAw(w, A).real
                return waw
            for (e, w) in zip(es, ws):
                if callable(ewindow):
                    if not ewindow(e):
                        continue
                if isinstance(operator, (list,)):
                    waws = [evaluate(w, k, A) for A in operator]
                else:
                    waws = [evaluate(w, k, operator)]
                out += str(k) + "   " + str(e) + "  "
                for waw in waws:
                    out += str(waw) + "  "
                out += "\n"
            if callback is not None:
                callback(k, es, ws)
        return out
    from pyqula import parallel
    if write:
        f = open(output_file, "w")
    esk = parallel.pcall(getek, range(len(kpath)))
    esk = "".join(esk)
    if write:
        f.write(esk)
        f.close()
    esk = esk.split("\n")
    del esk[-1]
    esk = np.array([[float(i) for i in ek.split()] for ek in esk]).T
    return esk


def test_get_bands_nd_matches_legacy_string_implementation(tmp_path):
    """The numeric rewrite of get_bands_nd must return the same data as the
    old implementation, which serialized every eigenvalue to a string and
    parsed it back out of a file."""
    os.chdir(tmp_path)
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    new = bandstructure.get_bands_nd(h, nk=12, write=True,
                                      output_file="BANDS_NEW.OUT")
    old = _legacy_get_bands_nd(h, nk=12, write=True,
                                output_file="BANDS_OLD.OUT")
    assert new.shape == old.shape
    assert np.allclose(new, old), "New and legacy bandstructures disagree"


def test_get_bands_nd_matches_legacy_with_operator_and_ewindow(tmp_path):
    """Same check, but exercising the operator-expectation-value branch and
    the energy-window filter, which change how many rows are emitted per
    k-point."""
    os.chdir(tmp_path)
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_zeeman([0., 0., 0.3])
    ewindow = lambda e: -1. < e < 1.
    new = bandstructure.get_bands_nd(h, nk=10, operator="sz",
                                      ewindow=ewindow, write=True,
                                      output_file="BANDS_NEW_OP.OUT")
    old = _legacy_get_bands_nd(h, nk=10, operator="sz",
                                ewindow=ewindow, write=True,
                                output_file="BANDS_OLD_OP.OUT")
    assert new.shape == old.shape
    assert np.allclose(new, old), \
        "New and legacy bandstructures (with operator) disagree"


def test_get_bands_nd_operator_list_fails_the_same_way_as_legacy(tmp_path):
    """operator=[op1, op2] (the shape examples/1d/several_operators/main.py
    uses) currently raises in *both* the legacy and the new implementation:
    get_bands_nd unconditionally calls h.get_operator(operator) on the raw
    argument before ever checking isinstance(operator, list), and
    operatorlist.get_operator has no list case, so it falls through to a
    bare `raise`. That's a pre-existing bug, not something introduced by
    the numba-batching refactor -- this test only pins down that the two
    implementations still fail identically, so the refactor doesn't change
    the failure mode."""
    os.chdir(tmp_path)
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_zeeman([0., 0., 0.3])
    ops = [h.get_operator("sz"), h.get_operator("sx")]
    new_exc = old_exc = None
    try:
        bandstructure.get_bands_nd(h, nk=8, operator=ops, write=False)
    except Exception as e:
        new_exc = type(e)
    try:
        _legacy_get_bands_nd(h, nk=8, operator=ops, write=False)
    except Exception as e:
        old_exc = type(e)
    assert new_exc is not None and new_exc == old_exc


def test_get_bands_nd_num_bands_eigenvalues_are_a_subset_of_the_spectrum(tmp_path):
    """num_bands routes through the arpack/sparse path, which this
    refactor left untouched except for moving hkgen's regeneration.
    arpack's Lanczos iteration uses a random start vector, so two
    independent calls (legacy vs. new) aren't guaranteed to return
    bit-identical results even with unchanged code -- checked directly:
    calling the *same* implementation twice already disagrees. Check
    something robust to that instead: every returned eigenvalue must be
    part of the true (fully diagonalized) spectrum."""
    os.chdir(tmp_path)
    g = geometry.honeycomb_lattice()
    g = g.get_supercell(3)
    h = g.get_hamiltonian()
    hkgen = h.get_hk_gen()
    kpath = h.geometry.get_kpath(None, nk=6)
    full_spectra = [np.linalg.eigvalsh(hkgen(k)) for k in kpath] # one per kpoint
    new = bandstructure.get_bands_nd(h, nk=6, num_bands=4, write=False)
    for ik, e in zip(new[0], new[1]):
        spectrum = full_spectra[int(round(ik))]
        assert np.min(np.abs(spectrum - e)) < 1e-6, \
            f"eigenvalue {e} at kpoint {ik} is not part of that kpoint's true spectrum"


def test_get_bands_nd_independent_of_batch_size(tmp_path):
    os.chdir(tmp_path)
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_zeeman([0., 0., 0.3])
    outs = [bandstructure.get_bands_nd(h, nk=10, operator="sz", write=False,
                                        batch_size=bs)
            for bs in (1, 3, 10, 100)]
    for o in outs[1:]:
        assert np.allclose(o, outs[0]), "Result depends on batch_size"


def test_get_bands_nd_callback_receives_every_kpoint_in_order():
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    seen = []
    bandstructure.get_bands_nd(h, nk=9, write=False,
                                callback=lambda k, es: seen.append(k))
    assert seen == list(range(9))
