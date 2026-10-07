# The truncated Chebyshev recursion of the KPM mean field
# (kpmtk/densitymatrix_kpm.py, future_development/sparse_kpm_mean_field.md).
#
# After n steps the vector started on orbital j is nonzero only within n
# hops of j, so the moments <e_i|T_n(H)|e_j> for i near j, computed on H
# restricted to the sites within R hops of j, are exact up to n = 2R with
# the doubling, and the recursion of every starting orbital costs the size
# of that ball instead of the size of the system. The starting orbitals are
# grouped in compact tiles, and every tile runs on one region, the tile and
# the rows of its pairs with every site within R hops of them; a tile is
# the width of a block of the recursion (pairmomentsnumba._MIN_COLUMNS), so
# the region is shared by as many starting orbitals as a block holds.
#
# The hops are those of the site graph, two sites joined whenever H(k) has
# an entry between their orbitals at some k-point, so that a spin flip or
# an electron-hole entry on one site is not a hop, and a bond across the
# cell of a periodic supercell is one. Beyond the light cone the moments
# are those of the restricted Hamiltonian, a finite cluster around the
# tile; the restriction to a region keeps every eigenvalue inside the
# spectrum of H (Cauchy interlacing), so the scale of the expansion stays
# valid. The error falls exponentially with R in a gapped state and at
# finite temperature, and does not fall in a metal at T=0.

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix


def check_radius(radius):
    """The radius of the truncation, a whole number of hops, or None"""
    if radius is None: return None
    if (isinstance(radius, (bool, np.bool_))
            or not isinstance(radius, (int, np.integer)) or radius < 0):
        raise ValueError("kpm_radius must be a whole number of hops, at "
                "least 0, or None for the full recursion, got %r" % (radius,))
    return int(radius)


def site_graph(ms, nsites):
    """The site graph of the matrices ms (N x N, one per k-point), N a
    multiple of nsites: a boolean CSR matrix joining two sites whenever
    some matrix has an entry between their orbitals"""
    N = ms[0].shape[0]
    if N % nsites != 0:
        raise ValueError("the truncated KPM recursion needs the same number "
                "of orbitals on every site, got %d orbitals on %d sites"
                % (N, nsites))
    per = N//nsites
    rows, cols = [], []
    for m in ms:
        c = coo_matrix(m)
        rows.append(c.row//per); cols.append(c.col//per)
    r, c = np.concatenate(rows), np.concatenate(cols)
    adj = csr_matrix((np.ones(len(r), dtype=bool), (r, c)),
            shape=(nsites, nsites))
    return (adj + adj.T).tocsr()


def tiles(positions, size):
    """Split the sites at positions into compact groups of at most size
    sites, cutting at the median along the longest side until every group
    fits; returns a list of index arrays"""
    positions = np.asarray(positions, dtype=float).reshape(len(positions), -1)
    out, todo = [], [np.arange(len(positions))]
    while todo:
        idx = todo.pop()
        if len(idx) <= size:
            out.append(idx)
            continue
        p = positions[idx]
        axis = np.argmax(p.max(axis=0) - p.min(axis=0))
        order = idx[np.argsort(p[:, axis], kind="stable")]
        todo += [order[:len(order)//2], order[len(order)//2:]]
    return out


def ball(adj, seeds, radius):
    """The sites within radius hops of the sites seeds, sorted"""
    reached = np.zeros(adj.shape[0], dtype=bool)
    frontier = np.unique(seeds)
    reached[frontier] = True
    for _ in range(radius):
        if len(frontier) == 0: break
        nb = np.unique(adj[frontier].indices)
        frontier = nb[~reached[nb]]
        reached[frontier] = True
    return np.nonzero(reached)[0]


# the tiles handed to the backend at once: their restricted matrices are
# built for one group at a time, so that the memory stays that of a group
# rather than of every tile, which at 4 x 10^5 orbitals was 9 GB
_TILES_PER_CALL = 512


# the regions of the last few calls: a mean-field loop asks for the same
# ones every iteration, since they depend only on the site graph, the pairs
# and the radius, and finding them was a sixth of an iteration at 10^5 sites
_REGIONS = dict()


def _key(*arrays):
    """A digest of the contents of arrays"""
    import hashlib
    d = hashlib.blake2b(digest_size=16)
    for a in arrays: d.update(np.ascontiguousarray(a).tobytes())
    return d.hexdigest()


def _tile_regions(ms, pairs, positions, radius, tile_size):
    """For every tile, the orbitals of its region, its pairs in the indexes
    of the region and the indexes of those pairs in pairs; and the largest
    region, block and number of pairs of any tile, which the card pads
    every tile to, so that its kernel compiles once. The result is kept
    for the next call with the same site graph, pairs, positions, radius
    and tiles"""
    N, nsites = ms[0].shape[0], len(positions)
    adj = site_graph(ms, nsites)
    key = (radius, tile_size, N, _key(adj.indptr, adj.indices, pairs,
        np.asarray(positions, dtype=float)))
    if key not in _REGIONS:
        if len(_REGIONS) >= 4: _REGIONS.clear()
        _REGIONS[key] = _find_regions(ms, adj, pairs, positions, radius,
                tile_size)
    return _REGIONS[key]


def _find_regions(ms, adj, pairs, positions, radius, tile_size):
    """_tile_regions, computed"""
    N, nsites = ms[0].shape[0], len(positions)
    per = N//nsites
    owner = pairs[:, 1]//per # a pair belongs to the site of its starting orbital
    order = np.argsort(owner, kind="stable")
    sorted_owner = owner[order]
    start_sites = np.unique(owner)
    regions = []
    width, npairs = 1, 1
    for tile in tiles(np.asarray(positions)[start_sites],
            max(1, tile_size//per)):
        tile = np.sort(start_sites[tile])
        lo = np.searchsorted(sorted_owner, tile, side="left")
        hi = np.searchsorted(sorted_owner, tile, side="right")
        sel = np.concatenate([order[a:b] for (a, b) in zip(lo, hi)])
        seeds = np.concatenate([tile, pairs[sel, 0]//per])
        sites = ball(adj, seeds, radius)
        orbitals = (sites[:, None]*per + np.arange(per)[None, :]).ravel()
        local = np.searchsorted(orbitals, pairs[sel])
        regions.append((orbitals, local, sel))
        width = max(width, len(np.union1d(local[:, 0], local[:, 1])))
        npairs = max(npairs, len(sel))
    union = abs(ms[0])
    for m in ms[1:]: union = union + abs(m)
    pad = dict(N=max(len(r[0]) for r in regions), width=width, npairs=npairs,
            K=int(np.max(np.diff(union.indptr))) if union.nnz else 1)
    return regions, pad


def pair_values(backend, ms, pairs, coef, positions, radius,
        kpm_prec="double", trace=False, tile_size=64):
    """The truncated counterpart of pair_values of the block recursion of
    backend (pairmomentsnumba or pairmomentsjax): the same values, with
    each tile of starting orbitals run on the Hamiltonian restricted to its
    region of radius hops. positions are those of the sites, and tile_size
    the orbitals of a tile. A backend with pair_values_batch takes the
    tiles in groups of _TILES_PER_CALL, each tile a member of one batch
    (one tile per thread on the CPU, and on the card every tile padded to
    the largest, so that the kernel compiles once whatever the sizes of
    the regions); otherwise the tiles go one by one"""
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    ms = [csr_matrix(m) for m in ms]
    out = np.zeros((len(ms), len(pairs)), dtype=np.complex128)
    tr = np.zeros((len(ms), len(coef)), dtype=np.complex128)
    mumax = 0.
    if len(pairs):
        regions, pad = _tile_regions(ms, pairs, positions, radius, tile_size)
        for g0 in range(0, len(regions), _TILES_PER_CALL):
            group = regions[g0:g0+_TILES_PER_CALL]
            problems = [([m[o][:, o] for m in ms], local)
                    for (o, local, _) in group]
            if hasattr(backend, "pair_values_batch"):
                results = backend.pair_values_batch(problems, coef,
                        kpm_prec=kpm_prec, pad=pad)
            else:
                results = [backend.pair_values(sub, local, coef,
                    kpm_prec=kpm_prec, trace=True) for (sub, local) in problems]
            for (_, _, sel), (vals, mm, t) in zip(group, results):
                out[:, sel] = vals
                tr += t
                mumax = max(mumax, mm)
    if trace: return out, mumax, tr
    return out, mumax


def trace_moments(backend, ms, nm, positions, radius, kpm_prec="double",
        tile_size=64):
    """The truncated counterpart of trace_moments of the block recursions:
    the first nm Chebyshev moments averaged over the orbitals, every
    orbital's from the region of radius hops around its tile"""
    N = ms[0].shape[0]
    pairs = np.stack([np.arange(N), np.arange(N)], axis=1)
    _, mumax, tr = pair_values(backend, ms, pairs, np.zeros(nm), positions,
            radius, kpm_prec=kpm_prec, trace=True, tile_size=tile_size)
    return tr/N, mumax
