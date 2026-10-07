# The local density of states of every orbital from a Chebyshev expansion,
# what an STM map measures, in a time linear in the number of sites.
#
# The local DOS of orbital i is the expansion of <e_i|delta(E-H)|e_i> in
# the moments <e_i|T_n(H)|e_i>, with the Jackson kernel, the expansion
# kpm.ldos makes for one site. Here the moments of every orbital come from
# the block recursion of the KPM mean field (kpmtk/pairmomentsnumba.py on
# the CPU, kpmtk/pairmomentsjax.py on the GPU), contracted at once with the
# Chebyshev basis at every energy asked for.
#
# 2*npol moments need T_n(H) e_i up to n=npol-1 only, with the doubling
# T_{2n} = 2 T_n T_n - T_0 and T_{2n+1} = 2 T_{n+1} T_n - T_1, and that
# vector is nonzero only within npol-1 hops of i. The moments computed on
# H restricted to the sites within npol-1 hops of the tile of i
# (kpmtk/truncation.py) are then exactly those of the whole system, so the
# map costs the number of sites times the size of that ball, which grows
# as npol^d: an energy resolution delta needs the environment of a site
# within roughly v/delta, whatever the size of the sample. A smaller
# kpm_radius trades that exactness for a finite cluster around each tile.

import numpy as np
from scipy.sparse import csr_matrix

from .kernels import jackson_kernel, jackson_npol


def kpm_ldos_basis(energies, scale, nm):
    """The (nm,ne) matrix that turns the Chebyshev moments mu_n of
    H/scale into the Jackson-broadened DOS at the energies,
    sum_n basis[n,e] mu_n, the same profile kpmtk.momenttoprofile gives
    divided by scale; zero outside (-scale,scale), where H/scale has no
    spectrum"""
    x = np.asarray(energies, dtype=float)/scale
    inside = np.abs(x) < 1.
    xin = np.where(inside, x, 0.)
    T = np.zeros((nm, len(x)))
    T[0] = 1.
    if nm > 1: T[1] = xin
    for n in range(2, nm): T[n] = 2.*xin*T[n-1] - T[n-2]
    c = np.full(nm, 2.); c[0] = 1. # mu_0 counts once, the others twice
    g = jackson_kernel(np.ones(nm)) # the Jackson damping factors
    w = np.where(inside, 1./(np.pi*scale*np.sqrt(1. - xin**2)), 0.)
    return (c*g)[:, None]*T*w[None, :]


def _operator_pairs(op, norb):
    """The pairs (j,i) of the entries A_ij of the matrix op, whose
    densities rho_ji give the operator-resolved local DOS Re(A rho)_ii,
    and the weights A_ij"""
    from scipy.sparse import coo_matrix
    a = coo_matrix(op)
    keep = a.data != 0.
    rows, cols, data = a.row[keep], a.col[keep], a.data[keep]
    return np.stack([cols, rows], axis=1).astype(np.int64), rows, data


def resolve_npol(scale, delta=None, npol=None):
    """The number of polynomials of the map: from the energy resolution
    delta (see kernels.jackson_npol), from npol, or the default of the
    KPM mean field when neither is given"""
    if delta is not None and npol is not None:
        raise ValueError("the KPM local DOS takes either the energy "
                "resolution delta or the number of polynomials npol, "
                "which sets it, not both; got delta=%r and npol=%r"
                % (delta, npol))
    if delta is not None: return jackson_npol(scale, delta)
    if npol is None:
        from .densitymatrix_kpm import DEFAULT_NPOL
        return DEFAULT_NPOL
    if (isinstance(npol, (bool, np.bool_))
            or not isinstance(npol, (int, np.integer)) or npol < 2):
        raise ValueError("npol must be a whole number of polynomials, at "
                "least 2, got %r" % (npol,))
    return int(npol)


def light_cone(npol):
    """The radius in hops within which the 2*npol moments of an orbital
    are exactly those of the whole system"""
    return max(npol - 1, 0)


def ldos_map(h, energies, ks, delta=None, npol=None, scale=None,
        kpm_radius=None, operator=None, kpm_prec=None, full=False):
    """Local DOS of every orbital of h at the energies, averaged over the
    k-points ks, from a Chebyshev expansion with the Jackson kernel.
    Returns an (ne,norb) array, each orbital's DOS integrating to one over
    the spectrum, and the Chebyshev moments of the DOS averaged over the
    orbitals and the k-points, with the scale they are of, so that a
    caller can draw the total DOS on another grid with kpm_ldos_basis.

    - delta: energy resolution, the half width at half maximum of the peak
      a single level gives (kernels.jackson_npol); or npol, the number of
      polynomials (2*npol moments); DEFAULT_NPOL when neither is given
    - scale: the spectrum of H(k) must lie inside [-scale,scale]; None
      estimates it from the Gershgorin bound
    - kpm_radius: None for the exact map, computed on the ball of the
      light cone around each tile; a smaller whole number of hops keeps a
      finite cluster around each tile, a larger one changes nothing
    - operator: None, or a k-independent matrix A (an Operator with a
      matrix), giving Re(A rho(E))_ii, the local matrix element, which is
      the convention of get_ldos with mode="green"
    - full: the recursion on the whole system for every orbital, whatever
      the radius, the reference the truncation is checked against"""
    from . import densitymatrix_kpm as dmk
    from . import truncation
    kpm_prec = dmk.resolve_kpm_prec(kpm_prec)
    hk_gen = h.get_hk_gen()
    ks = [np.array(k, dtype=float) for k in ks]
    given = scale is not None
    if scale is None: scale = dmk._estimate_kpm_scale(hk_gen, ks)
    if not np.isfinite(scale) or scale <= 0.:
        raise ValueError("the KPM local DOS needs a positive scale bounding "
                "the spectrum, got %r" % (scale,))
    npol = resolve_npol(scale, delta=delta, npol=npol)
    nm = 2*npol # the moments, as everywhere in the package
    radius = light_cone(npol)
    if kpm_radius is not None:
        radius = min(truncation.check_radius(kpm_radius), radius)
    if full: radius = None
    ms = [csr_matrix(hk_gen(k))/scale for k in ks]
    norb = ms[0].shape[0]
    diagonal = np.stack([np.arange(norb)]*2, axis=1).astype(np.int64)
    if operator is None:
        pairs = diagonal
    else:
        op = operator.get_matrix(required=False)
        if op is None:
            raise NotImplementedError("the KPM local DOS takes an operator "
                    "with a k-independent matrix; this one depends on k, "
                    "use mode='arpack'")
        if op.shape != (norb, norb):
            raise ValueError("the operator is a %dx%d matrix and the "
                    "Hamiltonian has %d orbitals" % (op.shape + (norb,)))
        opairs, orows, odata = _operator_pairs(op, norb)
        # the diagonal pairs too, for the trace of the DOS
        keys = np.union1d(opairs[:, 0]*norb + opairs[:, 1],
                np.arange(norb, dtype=np.int64)*(norb + 1))
        pairs = np.stack([keys//norb, keys % norb], axis=1)
        where = np.searchsorted(keys, opairs[:, 0]*norb + opairs[:, 1])
        weights = csr_matrix((odata, (orows, where)),
                shape=(norb, len(pairs))) # A_ij on the pair (j,i)
    basis = kpm_ldos_basis(energies, scale, nm)
    pair_values = dmk._pair_values_of(h, radius)
    # the k-points in groups, so that the contracted values of a group
    # stay below about 2^26 numbers whatever the mesh
    nkc = max(1, 2**26//max(1, len(pairs)*basis.shape[1]))
    ldos = np.zeros((basis.shape[1], norb))
    mus = np.zeros(nm)
    mumax = 0.
    for k0 in range(0, len(ms), nkc):
        vals, mm, tr = pair_values(ms[k0:k0+nkc], pairs, basis,
                kpm_prec=kpm_prec, trace=True)
        mumax = max(mumax, mm)
        mus += np.sum(tr.real, axis=0)
        for v in vals: # one k-point, (npairs,ne)
            if operator is None: ldos += v.real.T
            else: ldos += (weights@v).real.T
    dmk._check_scale_covers_spectrum([mumax], scale, given, kpm_prec=kpm_prec)
    return ldos/len(ms), mus/(len(ms)*norb), scale
