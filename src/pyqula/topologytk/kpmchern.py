"""Local Chern marker of a large flake from the KPM projector

The marker of realspace.real_space_chern needs the dense projector, a cost
that grows as the cube of the number of sites. Here the marker of a site a
is built only from the entries of the projector between the sites of its
ball, the sites within radius hops of a on the site graph, with every
position taken relative to a.

For a projector P (P^2=P, Q=1-P) the marker of an orbital alpha is

    c_alpha = 2*pi*Im [PXP,PYP]_aa = -2*pi*Im [PXQYP - PYQXP]_aa
            = 4*pi*Im sum_gamma (PXP)_ag (PYP)_ga

since PXP and PYP are Hermitian, so (PYPXP)_aa is the conjugate of
(PXPYP)_aa. Only the row alpha of PXP and of PYP enters,

    (PXP)_ag = sum_beta P_ab x_b P_bg,

and in the ball of a this is the row of P on the ball, times the
positions, times the block of P on the ball: P_ab with b in the ball, and
P_bg with b and g both in the ball, so within 2*radius hops of each other.
Those pairs are what get_dm_kpm_sparse computes, in memory linear in their
number. A site costs the square of the number of orbitals of its ball, the
pairs are the number of sites times the size of a ball of 2*radius hops,
and the recursion of each starting orbital runs on kpm_radius hops around
its tile (kpmtk/truncation.py). None of these depends on the size of the
flake, so at a fixed radius the cost is linear in the number of sites. The
sites go in compact groups of _SITES_PER_CALL, each with its own call of
the density matrix, so that the memory is that of a group; the pairs of
the halo of a group, radius hops wide, are computed twice.

The positions are relative to a because the KPM projector at zero
temperature is not idempotent. A shift X -> X - x0 adds to [PXP,PYP] the
terms -x0 [P^2,PYP] - y0 [PXP,P^2], which vanish when P^2=P, and which for
the expanded projector are its error multiplied by the distance to the
origin. With every site its own origin the shift changes nothing in the
exact marker and keeps that error local.

The sum rule is lost. The dense marker of an open flake sums to zero, as
the trace of a commutator of finite matrices, with the large negative
marker of the edge compensating the bulk. The edge states are gapless, so
the projector decays slowly along the edge, and a ball of finite radius
does not see that compensation: the local marker of the exact projector
on the same balls sums to about 0.85 times the number of sites on a
Haldane flake of 984 sites at radius 4 to 8, and the KPM one to the same.
This marker is for the bulk, where the gap is open.

The normalization, the sign and the operator= of real_space_chern are kept,
and the value of each site is divided by its Voronoi area (site_areas).
With the exact projector and balls that cover the flake the marker is the
dense one to roundoff, for operator=None and for an operator that commutes
with P. For one that does not, sz with Rashba coupling, the operator
marker of real_space_chern itself depends on the origin of the positions,
by terms like x0*Im(P op PYP - PYP op P)_aa, and this one takes each site
as its origin: on a Kane-Mele flake of 104 sites with a Rashba coupling of
0.1 the two differ by up to 0.27, against 2.6e-14 without it.
"""
import numpy as np
from numba import jit
from scipy.sparse import csr_matrix, identity


def _check_count(name, x, least):
    """x, a whole number at least least"""
    if (isinstance(x, (bool, np.bool_))
            or not isinstance(x, (int, np.integer)) or x < least):
        raise ValueError("%s must be a whole number, at least %d, got %r"
                % (name, least, x))
    return int(x)


def _reach(adj, radius):
    """Boolean CSR matrix of the pairs of sites within radius hops on the
    site graph adj, with sorted indices"""
    n = adj.shape[0]
    step = (adj + identity(n, dtype=bool, format="csr")).astype(bool).tocsr()
    out = identity(n, dtype=bool, format="csr")
    for _ in range(radius):
        out = (out @ step).astype(bool).tocsr()
    out.sort_indices()
    return out


def _orbital_pairs(reach, per):
    """The orbital pairs (rows, cols) of the site pairs of reach, every
    orbital of one site with every orbital of the other"""
    c = reach.tocoo()
    o = np.arange(per)
    rows = (c.row[:, None, None]*per + o[None, :, None]
            + 0*o[None, None, :]).ravel()
    cols = (c.col[:, None, None]*per + o[None, None, :]
            + 0*o[None, :, None]).ravel()
    return rows.astype(np.int64), cols.astype(np.int64)


@jit(nopython=True, cache=True)
def _block(indptr, indices, data, J, where):
    """The dense block of the CSR matrix (indptr, indices, data) on the
    orbitals J; where is -1 on every orbital and is left so"""
    n = len(J)
    out = np.zeros((n, n), dtype=data.dtype)
    for k in range(n): where[J[k]] = k
    for k in range(n):
        i = J[k]
        for p in range(indptr[i], indptr[i+1]):
            l = where[indices[p]]
            if l >= 0: out[k, l] = data[p]
    for k in range(n): where[J[k]] = -1
    return out


def local_markers(P, ball, positions, per, op=None, sites=None):
    """Marker of the sites (every site if None), times its area, from the
    projector P (CSR, norb x norb, the entries between the sites of each
    ball are enough), ball the CSR matrix of the sites within the radius
    of each site, positions the xy of the sites and per the orbitals of a
    site. op is None or the CSR matrix of the operator of
    real_space_chern"""
    if sites is None: sites = np.arange(ball.shape[0])
    P = csr_matrix(P, dtype=np.complex128)
    P.sort_indices()
    where = -np.ones(P.shape[0], dtype=np.int64)
    if op is not None:
        op = csr_matrix(op, dtype=np.complex128)
        op.sort_indices()
    orb = np.arange(per)
    out = np.zeros(len(sites))
    for (n, a) in enumerate(sites):
        near = ball.indices[ball.indptr[a]:ball.indptr[a+1]]
        J = np.sort((near[:, None]*per + orb[None, :]).ravel())
        Pb = _block(P.indptr, P.indices, P.data, J, where) # P on the ball
        d = np.repeat(positions[np.sort(near)] - positions[a], per, axis=0)
        x, y = d[:, 0], d[:, 1] # relative to site a
        ia = np.searchsorted(J, a*per + orb) # the orbitals of a in the ball
        Pa = Pb[ia, :] # the rows of P of the orbitals of a
        A = (Pa*x[None, :]) @ Pb # rows of PXP
        B = (Pa*y[None, :]) @ Pb # rows of PYP
        if op is None: # 2*Im (AB)_aa, with B Hermitian
            c = 2.*np.sum(A*B.conj(), axis=1)
        else: # (A op B - B A op + A B op - B op A)/2, as real_space_chern
            o = _block(op.indptr, op.indices, op.data, J, where)
            oa = o[:, ia] # columns of op of the orbitals of a
            BA = ((B @ Pb)*x[None, :]) @ Pb # rows of B A
            AB = ((A @ Pb)*y[None, :]) @ Pb # rows of A B
            c = (np.sum((A @ o)*B.conj(), axis=1)
                    - np.sum(BA*oa.T, axis=1)
                    + np.sum(AB*oa.T, axis=1)
                    - np.sum((B @ o)*A.conj(), axis=1))/2.
        out[n] = np.sum(c.imag)
    return 2.*np.pi*out


# the sites whose marker one call of the KPM density matrix serves: the
# projector of a group lives in memory only while its markers are taken,
# so that the memory is that of a group, the pairs of all the sites at
# radius 6 being 6e7 at 1e5 sites
_SITES_PER_CALL = 4096


def kpm_chern_marker(h, radius=6, npol=None, kpm_radius=None,
        operator=None, scale=None, kpm_prec=None):
    """Local Chern marker of a finite (0d) Hamiltonian from the KPM
    projector, one value per site, at a cost linear in the number of
    sites. The occupied states are those below zero energy, as in
    real_space_chern, and each value is divided by the area of the site.
    Returns the positions and the marker.

    - radius: the ball of each site, in hops on the site graph; in a
      gapped state the marker converges exponentially with it
    - npol: number of Chebyshev polynomials (2*npol moments), the default
      of the KPM mean field if None; it sets how sharp the step of the
      projector is, which only has to fit inside the gap
    - kpm_radius: truncation of the Chebyshev recursion in hops
      (kpmtk/truncation.py), radius if None
    - operator: restrict the marker to a subspace, as real_space_chern
    - scale: bound of the spectrum, estimated if None; a given one has to
      exceed the spectral radius by more than 1% (see densitymatrix_kpm)
    """
    from ..kpmtk import densitymatrix_kpm as dmk
    from ..kpmtk import truncation
    from .realspace import site_areas
    if h.dimensionality != 0:
        raise ValueError("the KPM Chern marker needs a finite (0d) "
                "Hamiltonian, a flake, got dimensionality %r"
                % (h.dimensionality,))
    if not h.is_sparse:
        # a dense intra costs a pass over its N^2 entries in every group
        # (measured 22 s of 38 at 1e4 sites), so the cost is no longer
        # linear; one sparse copy here removes it
        h = h.copy()
        h.turn_sparse()
    radius = _check_count("radius", radius, 1)
    if npol is None: npol = dmk.DEFAULT_NPOL
    npol = _check_count("npol", npol, 2)
    if kpm_radius is None: kpm_radius = radius
    kpm_radius = truncation.check_radius(kpm_radius)
    op = None
    if operator is not None:
        if not hasattr(operator, "get_matrix"):
            raise TypeError("the projector must be an operator with a "
                    "matrix representation, or None, got %r"
                    % (type(operator).__name__,))
        op = csr_matrix(operator.get_matrix())
    h0 = csr_matrix(h.intra)
    nsites = len(h.geometry.r)
    adj = truncation.site_graph([h0], nsites) # refuses unequal sites
    per = h0.shape[0]//nsites
    if op is not None and op.shape != h0.shape:
        raise ValueError("the operator is a %dx%d matrix and the "
                "Hamiltonian has %d orbitals" % (op.shape + (h0.shape[0],)))
    ball = _reach(adj, radius)
    positions = np.array(h.geometry.r)[:, 0:2]
    C = np.zeros(nsites)
    for group in truncation.tiles(positions, _SITES_PER_CALL):
        group = np.sort(group)
        # the pairs of sites that share the ball of a site of the group
        share = ball[group].T.astype(np.int32) @ ball[group].astype(np.int32)
        rows, cols = _orbital_pairs(share.astype(bool).tocsr(), per)
        dm = dmk.get_dm_kpm_sparse(h, {(0, 0, 0): (rows, cols)}, nk=1,
                scale=scale, npol=npol, kpm_prec=kpm_prec,
                radius=kpm_radius)
        P = dm[(0, 0, 0)].T.tocsr() # dm[i,j] = <c^dag_i c_j> is P transposed
        C[group] = local_markers(P, ball, positions, per, op=op, sites=group)
    return h.geometry.r, C/site_areas(h.geometry)
