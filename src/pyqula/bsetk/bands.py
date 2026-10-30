import numpy as np


def exciton_bands(h,qpath=None,nq=None,n=None,**kwargs):
    """Return the exciton bands: the Bethe-Salpeter energies at finite
    center-of-mass momentum Q, scanned along a q-path.

    An exciton is a two-particle state, so it has a dispersion of its own:
    the whole electron-hole pair can propagate with momentum Q, and its
    energy E_X(Q) is what this returns. That dispersion is not the
    difference of two band energies -- it is bent by the electron-hole
    interaction, and its curvature is the exciton's effective mass, so a
    heavy, flat exciton band is a strongly bound (small, tightly localized)
    exciton and a steep one a weakly bound Wannier-Mott exciton.

    One full BSE is solved per q-point (a dense diagonalization each), so
    the cost is nq times the cost of a single get_bse call -- keep the pair
    basis small via nv/nc and use tda=True when only the lowest excitons
    matter (n only selects how many of the computed energies come back).

    Q is not restricted to the k-mesh: the pair basis diagonalizes at k and
    k+Q independently, so any q-path works at any nk.

    qpath/nq select the path (same convention as get_bands/get_magnon_bands,
    so a list of high-symmetry labels or of explicit q-vectors also works),
    n keeps only the n lowest excitons at each q-point, and every other
    argument (V, nk, nv, nc, kernel, tda, max_memory, screening, nkW) is
    passed on to get_bse unchanged.

    The screened interaction does not depend on Q, so screening="rpa" or
    "crpa" builds it once, on the mesh of the first q-point, and every
    q-point of the path reuses it.

    Returns (qs,es): qs is the integer index of the q-point along the path
    (the same convention get_bands uses for its k-axis) and es the exciton
    energy, both flat 1D arrays of equal length ready for a scatter-style
    dispersion plot. es comes back complex if the mean-field reference is
    unstable against some excitation at some q-point -- a sizable imaginary
    part is physically meaningful there, and is reported rather than
    silently dropped, exactly as in a single BSE solve."""
    from .solve import BSE
    from .. import parallel
    if h.dimensionality<1:
        raise ValueError("exciton bands need a periodic Hamiltonian, a 0d "
                "system has no center-of-mass momentum to disperse in")
    from ..klist import get_qpath
    qpath = get_qpath(h.geometry,qpath,nq) # generate the q-path
    if n is not None and kwargs.get("solver","dense")=="iterative":
        # the iterative solver returns neig excitons, 4 by default, which
        # n would otherwise only truncate
        kwargs.setdefault("neig",n)
    screening = kwargs.get("screening")
    if isinstance(screening,str) and kwargs.get("solver","dense")=="dense":
        # one W for the whole path, as spectral._qpath_map does; it used to
        # be rebuilt at every q-point. The other solvers refuse or handle
        # screening on their own, and keep doing so
        from .pairbasis import PairBasis
        from .solve import get_direct_interaction
        nk = kwargs.get("nk",10)
        pb = PairBasis(h,Q=qpath[0],nk=nk,nv=kwargs.get("nv"),
                nc=kwargs.get("nc"))
        kwargs["screening"] = get_direct_interaction(h,pb,V=kwargs.get("V"),
                nk=nk,screening=screening,nkW=kwargs.get("nkW"),
                kernel=kwargs.get("kernel","full"),
                channel=kwargs.get("channel","charge"))
    def f(q):
        return BSE(h,Q=q,**kwargs).get_energies(n=n)
    outs = parallel.pcall(f,qpath) # solve one BSE per q-point
    qs = np.concatenate([np.full(len(es),iq) for iq,es in enumerate(outs)])
    es = np.concatenate(outs)
    return qs,es
