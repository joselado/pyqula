import numpy as np
from scipy.sparse import csc_matrix

# default onsite strength used to emulate a vacancy: a real site removal
# would require deleting rows/columns from every sparse intra/inter block of
# a multicell Hamiltonian, which is not implemented here; a strong onsite
# potential pushes the site's level far outside the energy window of
# interest instead, which is the standard tight-binding proxy for a vacancy
# and keeps the Hamiltonian's sparsity pattern and site bookkeeping untouched
default_vacancy_strength = 100.0


def _normalize_nsuper(nsuper):
    """Turn a scalar or (n1,n2) pair into the explicit (n1,n2,1) triple
    multicell.supercell_hamiltonian expects. h.get_supercell only pads a
    *scalar* nsuper this way; passed through as-is, a 2-tuple crashes
    with an IndexError deep in multicell.py (it unconditionally unpacks
    3 elements) and a 3-tuple is silently inconsistent between
    geometry.get_supercell (which only reads n1,n2 for a 2D lattice) and
    multicell.supercell_hamiltonian (which sizes the Hamiltonian by
    n1*n2*n3 regardless of dimensionality) -- doing the padding once
    here, explicitly, avoids both."""
    try: n1,n2 = nsuper[0],nsuper[1]
    except (TypeError,IndexError): n1 = n2 = nsuper
    return (int(n1),int(n2),1)


def _impurity_onsite(imp):
    """Return the onsite strength for one impurity spec, requiring
    exactly one of 'onsite'/'vacancy' -- a spec with neither (e.g. a
    typo'd key) used to silently default to onsite=0.0, a no-op that
    looked like a valid impurity."""
    has_onsite = "onsite" in imp
    has_vacancy = imp.get("vacancy",False)
    if has_onsite == bool(has_vacancy):
        raise ValueError("impurity spec needs exactly one of 'onsite' or 'vacancy': "+str(imp))
    if has_vacancy: return imp.get("strength",default_vacancy_strength)
    return imp["onsite"]


def build_impurity_hamiltonian(h,nsuper,impurities):
    """Build a supercell of h with a list of real-space impurities added
    as onsite potentials.

    nsuper: supercell size, scalar or (n1,n2)
    impurities: list of dicts, each specifying where the impurity sits,
      either 'index' (site index in the supercell geometry) or 'position'
      (a real-space coordinate, snapped to the closest actual site), and
      how strong it is, either 'onsite' (a number) or 'vacancy'=True
      (uses default_vacancy_strength unless 'strength' is also given)

    The Hamiltonian is kept sparse throughout: h.turn_sparse() is applied
    before the supercell is built (sparse=True, already get_supercell's
    default) so the large supercell block matrix is never densified, and
    the onsite potential is added as a sparse diagonal built directly
    from the resolved site indices, so this never allocates a dense
    matrix of the supercell's size, nor a per-site Python callable
    (hamiltonians.add_onsite's generic position-function path) when the
    exact indices are already known here."""
    h = h.copy()
    h.turn_sparse()
    hs = h.get_supercell(_normalize_nsuper(nsuper),sparse=True)
    hs.turn_sparse()
    if len(impurities)==0: return hs
    g = hs.geometry
    n = len(g.r)
    onsite = np.zeros(n,dtype=np.complex128)
    for imp in impurities:
        if "index" in imp: i = imp["index"]
        elif "position" in imp: i = g.closest_index(np.array(imp["position"]))
        else: raise ValueError("impurity spec needs 'index' or 'position': "+str(imp))
        onsite[i] += _impurity_onsite(imp)
    idx = np.arange(n)
    m = csc_matrix((onsite,(idx,idx)),shape=(n,n),dtype=np.complex128)
    hs.intra = hs.intra + hs.spinless2full(m)
    return hs
