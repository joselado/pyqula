# Sparse counterparts of the pieces of the density-density mean field
# (scftk/densitydensity.py, scftk/superscf.py) that the KPM engine goes
# through every iteration, for a Hamiltonian too large to hold an n x n
# matrix: the interaction, the density matrix and the mean field are
# dictionaries {direction: csr_matrix} holding only the entries the
# interaction couples, and each step costs time and memory linear in their
# number. densitydensity_kpm.py routes a sparse Hamiltonian (h.is_sparse)
# here and a dense one to the dense functions, which stay as they are; each
# function below names the dense one it mirrors, and
# tests/scf/test_sparse_kpm_mean_field.py holds the two to each other.
import itertools

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, diags
from scipy.spatial import cKDTree


def _minus(d): return (-d[0], -d[1], -d[2])


def _entries(m, rows, cols):
    """The entries (rows[k],cols[k]) of a sparse matrix, zero where m stores
    nothing"""
    if len(rows) == 0: return np.zeros(0, dtype=np.complex128)
    return np.asarray(csr_matrix(m)[rows, cols]).ravel()


def site_pairs(g, rmax):
    """Every pair of sites (i, j in the cell at R) at a distance up to rmax,
    found with a KD-tree, as a dictionary {R: (i, j, squared distance)}
    over the cells R that hold at least one; (0,0,0) is always there, and
    the pair of a site with itself is one of its pairs. Only the cells
    whose bounding box comes within rmax of the unit cell's are searched."""
    r = np.array(g.r, dtype=float)
    tree = cKDTree(r)
    lo, hi = np.min(r, axis=0), np.max(r, axis=0)
    dirs = [(0, 0, 0)]
    if g.dimensionality > 0:
        nd = g.dimensionality
        B = np.linalg.pinv(np.array([g.a1, g.a2, g.a3][:nd])) # R@B, cell of R
        span = np.linalg.norm(hi - lo) # no pair in the cell is longer
        nmax = int(np.ceil((rmax + span)*np.max(np.linalg.norm(B, axis=0)))) + 1
        dirs = [tuple(list(c) + [0]*(3 - nd))
                for c in itertools.product(range(-nmax, nmax + 1), repeat=nd)]
    out = dict()
    for d in dirs:
        R = d[0]*np.array(g.a1) + d[1]*np.array(g.a2) + d[2]*np.array(g.a3)
        gap = np.maximum(0., np.maximum(lo + R - hi, lo - hi - R))
        if np.linalg.norm(gap) > rmax and d != (0, 0, 0): continue # too far
        near = tree.query_ball_point(r + R, rmax) # the i near each j
        j = np.repeat(np.arange(len(r)), [len(x) for x in near])
        i = np.array([ii for x in near for ii in x], dtype=np.int64)
        dr = r[j] + R - r[i]
        d2 = dr[:, 0]*dr[:, 0] + dr[:, 1]*dr[:, 1] + dr[:, 2]*dr[:, 2]
        if len(i) > 0 or d == (0, 0, 0): out[d] = (i, j, d2)
    return out


# the weights of the four spin entries (up-up, up-down, down-up,
# down-down) of a pair: a density-density interaction couples the two
# densities whatever the spins, and Sz_i Sz_j = (n_iu-n_id)(n_ju-n_jd)/4
DENSITY = np.array([[1., 1.], [1., 1.]])
SZSZ = np.array([[1., -1.], [-1., 1.]])/4.


def interaction(h, V1=0.0, V2=0.0, V3=0.0, U=None, Vr=None, rcut=None,
        spin=DENSITY, name="Vr"):
    """Sparse counterpart of the interaction dictionary that
    Vinteraction_kpm builds: V1/V2/V3 halved on the first three
    neighbor shells of g.neighbor_distances(), matched on the squared
    distance to 1e-4 as specialhopping.distance_hopping_matrix does, Vr
    halved on every pair up to rcut with the pair of a site with itself
    included (densitydensity.add_pair_interaction), each on the four spin
    entries of a pair of a spinful Hamiltonian weighted by spin (DENSITY,
    or SZSZ for the exchange channels of spinspin._build_v, where V1/V2/V3
    and Vr stand for J1/J2/J3 and Jr, named by name in a message), and U,
    an array over the sites, halved on the two up-down entries of each
    site. The pairs come from a KD-tree instead of evaluating every pair of
    sites.

    Two differences with the dense builder, both where it truncates: the
    shells reach every cell that holds one, while the dense one only looks
    two cells away, and Vr on a finite system needs an explicit rcut, since
    rcut=None means every pair there, which is N^2 bonds."""
    g = h.geometry
    n = len(g.r) # number of sites
    rows, cols, vals = dict(), dict(), dict()
    def add(d, i, j, x):
        keep = x != 0.
        if not np.any(keep) and d != (0, 0, 0): return # a direction without
        rows.setdefault(d, []).append(i[keep])
        cols.setdefault(d, []).append(j[keep])
        vals.setdefault(d, []).append(np.asarray(x, dtype=np.complex128)[keep])
    vs = np.array([V1/2., V2/2., V3/2.], dtype=np.complex128)
    if np.any(vs != 0.):
        ds2 = np.array(g.neighbor_distances())[0:3]**2 # the shells
        vs = vs[0:len(ds2)]
        for d, (i, j, d2) in site_pairs(g, np.sqrt(np.max(ds2) + 1e-3)).items():
            x = np.zeros(len(i), dtype=np.complex128)
            for k in range(len(ds2)): # a later shell wins, as in the dense
                x[np.abs(ds2[k] - d2) < 1e-4] = vs[k]
            add(d, i, j, x)
    if Vr is not None:
        if not callable(Vr):
            raise TypeError(name+" must be a function "+name+"(r1,r2) of "
                    "two positions, got "+str(type(Vr)))
        if rcut is None:
            if g.dimensionality == 0:
                raise ValueError(name+" on a finite system with rcut=None "
                        "means every pair of sites, which is N^2 bonds and "
                        "defeats the sparse mean field; pass rcut, the "
                        "distance beyond which "+name+" is dropped")
            rcut = 5.0 # as specialhopping.distance_cut_interaction
        tol = 1e-6 # a shell sitting exactly at rcut is kept whole
        r = np.array(g.r, dtype=float)
        for d, (i, j, d2) in site_pairs(g, float(rcut) + tol).items():
            R = d[0]*np.array(g.a1) + d[1]*np.array(g.a2) + d[2]*np.array(g.a3)
            x = np.array([Vr(r[a], r[b] + R) for (a, b) in zip(i, j)],
                    dtype=np.complex128)/2.
            add(d, i, j, x)
    if (0, 0, 0) not in rows: add((0, 0, 0), np.zeros(0, int), np.zeros(0, int),
            np.zeros(0))
    s = 2 if h.has_spin else 1 # spin entries per site
    v = dict()
    for d in rows:
        i, j = np.concatenate(rows[d]), np.concatenate(cols[d])
        x = np.concatenate(vals[d])
        if h.has_spin: # the four spin entries of each pair
            a, b = np.meshgrid([0, 1], [0, 1], indexing="ij")
            i = (2*i[:, None] + a.ravel()[None, :]).ravel()
            j = (2*j[:, None] + b.ravel()[None, :]).ravel()
            x = (x[:, None]*np.asarray(spin).ravel()[None, :]).ravel()
        v[d] = csr_matrix((x, (i, j)), shape=(s*n, s*n)) # sums duplicates
    if U is not None and np.max(np.abs(U)) > 0.:
        i = np.arange(n)
        u = np.asarray(U, dtype=np.complex128)/2.
        v[(0, 0, 0)] = v[(0, 0, 0)] + csr_matrix((np.concatenate([u, u]),
            (np.concatenate([2*i, 2*i + 1]), np.concatenate([2*i + 1, 2*i]))),
            shape=(s*n, s*n))
    return v


def needed_entries(v, n, has_eh=False, blocks=False, tol=1e-10):
    """Sparse counterpart of densitymatrix_kpm.required_elements (and
    required_elements_eh for a Nambu density matrix of dimension n): the
    entries of the density matrix the mean field and its double counting
    read, as {direction: (rows, cols)} index arrays. Per entry v[d][a,b]:
    dm[d][a,b] (the double counting), dm[-d][b,a] (the Fock term) and the
    occupations dm[0][a,a] and dm[0][b,b] (the Hartree term); with Nambu,
    each of these at the electron positions of the Nambu basis, and the
    pairing entry dm[-d] between the electron b^1 and the hole a
    (superscf.anomalous_term_ij_jit). The raw, unmapped positions that
    required_elements_eh also asks for are left out, since the KPM mean
    field reads the double counting off the electron sector.

    v can also be a list of interaction dictionaries, the channels of
    spinspin.VJinteraction, whose entries are then all needed. blocks=True
    completes every entry to its 2 x 2 block of the index pairs (2k,2k+1),
    the spin of a site (or, with Nambu, its electrons and its holes), which
    a channel decoupled in a rotated spin frame needs: the rotation mixes
    the two indexes of every such block (spinspin._block_rotate), so a
    rotated entry is exact only where its whole block was computed."""
    if isinstance(v, dict): v = [v]
    out = dict()
    def add(d, r, c): out.setdefault(d, []).append((r, c))
    if has_eh:
        def e(x): return 4*(x//2) + x % 2 # electron of orbital x
        def hole(x): return 4*(x//2) + 2 + x % 2 # its hole partner
    else:
        def e(x): return x
    for (d, m) in [(d, m) for vi in v for (d, m) in vi.items()]:
        m = coo_matrix(m)
        keep = np.abs(m.data) > tol
        a, b = m.row[keep].astype(np.int64), m.col[keep].astype(np.int64)
        add(d, e(a), e(b))
        add(_minus(d), e(b), e(a))
        add((0, 0, 0), e(a), e(a))
        add((0, 0, 0), e(b), e(b))
        if has_eh: add(_minus(d), e(b ^ 1), hole(a))
    needed = dict()
    for d, lists in out.items():
        r = np.concatenate([r for (r, c) in lists])
        c = np.concatenate([c for (r, c) in lists])
        if blocks: # every entry of the 2 x 2 block of each one
            s1, s2 = np.meshgrid([0, 1], [0, 1], indexing="ij")
            r = (2*(r//2)[:, None] + s1.ravel()[None, :]).ravel()
            c = (2*(c//2)[:, None] + s2.ravel()[None, :]).ravel()
        key = np.unique(r*n + c)
        needed[d] = (key//n, key % n)
    return needed


def get_mf_normal(v, dm, compute_dd=True, add_dagger=True, compute_cross=True):
    """Sparse counterpart of densitydensity.get_mf_normal (through
    get_mf_normal_core and the normal_term_*_jit loops): the Fock term
    -v[d][i,j] dm[-d][j,i] on the entries of v[d], with its Hermitian
    conjugate at -d, and the Hartree terms v[d] n and v[-d]^T n on the
    diagonal, n the occupations dm[0][i,i]"""
    n = dm[(0, 0, 0)].shape[0]
    occ = dm[(0, 0, 0)].diagonal()
    mf = {d: csr_matrix((n, n), dtype=np.complex128) for d in v}
    for d in v:
        d2 = _minus(d)
        if compute_cross:
            m = coo_matrix(v[d])
            x = -m.data*_entries(dm[d2], m.col, m.row)
            t = csr_matrix((x, (m.row, m.col)), shape=(n, n))
            mf[d] = mf[d] + t
            if add_dagger: mf[d2] = mf[d2] + t.conj().T.tocsr()
        if compute_dd:
            hartree = v[d] @ occ + v[d2].T @ occ
            mf[(0, 0, 0)] = mf[(0, 0, 0)] + diags(hartree, format="csr")
    return mf


def get_mf_anomalous(v, dm):
    """Sparse counterpart of superscf.get_mf_anomalous, dm the electron-hole
    sector: the pairing term 2 v[d][a,b] dm[-d][b^1,a] at (a,b^1) for every
    entry of v[d], the one rule that the four spin cases of
    anomalous_term_ij_jit reduce to"""
    n = dm[(0, 0, 0)].shape[0]
    mf = dict()
    for d in v:
        m = coo_matrix(v[d])
        a, b = m.row.astype(np.int64), m.col.astype(np.int64)
        x = 2*m.data*_entries(dm[_minus(d)], b ^ 1, a)
        mf[d] = csr_matrix((x, (a, b ^ 1)), shape=(n, n))
    return mf


def enforce_eh_symmetry_anomalous(d01):
    """Sparse counterpart of superscf.enforce_eh_symmetry_anomalous: the
    (0,1) sector averaged with its electron-hole image, entry (a,b) with
    s(a,b) d[-R][b^1,a^1], s=+1 when a and b have the same spin and -1
    otherwise (the four cases of enforce_eh_symmetry_anomalous_jit), and
    the (1,0) sector at -R as its Hermitian conjugate"""
    n = next(iter(d01.values())).shape[0]
    z = diags(1. - 2.*(np.arange(n) % 2), format="csr") # +1 up, -1 down
    flip = csr_matrix((np.ones(n), (np.arange(n), np.arange(n) ^ 1)),
            shape=(n, n)) # a -> a^1
    out01, out10 = dict(), dict()
    for key in d01:
        image = z @ flip @ d01[_minus(key)].T @ flip @ z
        out01[key] = ((d01[key] + image)/2.).tocsr()
    for key in out01: out10[_minus(key)] = out01[key].conj().T.tocsr()
    return out01, out10


def get_mf_bdg(v, dm, compute_anomalous=True, compute_normal=True, **kwargs):
    """Sparse counterpart of superscf.get_mf_bdg: the normal decoupling of
    the electron sector and the pairing decoupling of the electron-hole
    sector, put together in the Nambu basis"""
    from ..superconductivity import get_eh_sector, build_nambu_matrix
    from ..multihopping import MultiHopping
    dme = {key: get_eh_sector(m, i=0, j=0) for (key, m) in dm.items()}
    dma = {key: get_eh_sector(m, i=0, j=1) for (key, m) in dm.items()}
    mfe = get_mf_normal(v, dme, **kwargs)
    mfa01, mfa10 = enforce_eh_symmetry_anomalous(get_mf_anomalous(v, dma))
    mf = dict()
    for key in v:
        e = mfe[key] if compute_normal else mfe[key]*0.
        if compute_anomalous:
            mf[key] = build_nambu_matrix(e, c12=mfa10[key], c21=mfa01[key])
        else: mf[key] = build_nambu_matrix(e)
    if not MultiHopping(mf).is_hermitian(): # sanity check on the result
        raise ValueError("the BdG mean field came out non-Hermitian, which "
                "means a non-Hermitian density matrix or interaction")
    return mf


def get_mf(v, dm, has_eh=False, compute_anomalous=True, compute_normal=True,
        **kwargs):
    """Sparse counterpart of densitydensity.get_mf"""
    if has_eh:
        return get_mf_bdg(v, dm, compute_anomalous=compute_anomalous,
                compute_normal=compute_normal, **kwargs)
    return get_mf_normal(v, dm, **kwargs)


def get_dc_energy(v, dm):
    """Sparse counterpart of densitydensity.get_dc_energy (through
    get_dc_energy_jit): -v[d][i,j] n_i n_j + v[d][i,j] |dm[d][i,j]|^2
    summed over the entries of v"""
    occ = dm[(0, 0, 0)].diagonal()
    out = 0.0
    for d in v:
        m = coo_matrix(v[d])
        out -= np.sum(m.data*occ[m.row]*occ[m.col])
        c = _entries(dm[d], m.row, m.col) if d in dm else 0.*m.data
        out += np.sum(m.data*c*np.conjugate(c))
    return np.real(out)


def get_dc_energy_anomalous(mf, dm):
    """Sparse counterpart of superscf.get_dc_energy_anomalous, minus half
    the expectation value of the pairing part of the mean field"""
    from ..superconductivity import get_eh_sector
    out = 0.0
    for d in mf:
        m = get_eh_sector(mf[d], i=0, j=1)
        if m.nnz == 0 or d not in dm: continue
        out += get_eh_sector(dm[d], i=0, j=1).multiply(m).sum()
    return -np.real(out)/2.


def random_guess(v, h, scale=1.0):
    """Sparse counterpart of densitydensity.random_hermitian_guess: random
    phases on every entry of the blocks of the pairs of sites the
    interaction couples, and of every site with itself, in the Hilbert
    space of h (so with Nambu the pairing is seeded too), mirrored at -d
    and made Hermitian onsite, rather than on every entry of an n x n
    matrix"""
    ns = len(h.geometry.r) # number of sites
    b = h.intra.shape[0]//ns # orbitals per site of h
    bv = v[(0, 0, 0)].shape[0]//ns # orbitals per site of v
    a = np.arange(b)
    mf = dict()
    for d in v:
        if _minus(d) in mf: # mirror the opposite direction
            mf[d] = mf[_minus(d)].conj().T.tocsr() ; continue
        m = coo_matrix(v[d])
        si, sj = m.row//bv, m.col//bv # the pairs of sites
        if d == (0, 0, 0):
            si = np.concatenate([si, np.arange(ns)])
            sj = np.concatenate([sj, np.arange(ns)])
        key = np.unique(si.astype(np.int64)*ns + sj)
        si, sj = key//ns, key % ns
        rows = (b*si[:, None, None] + a[None, :, None] + 0*a[None, None, :]).ravel()
        cols = (b*sj[:, None, None] + 0*a[None, :, None] + a[None, None, :]).ravel()
        x = np.exp(1j*np.random.random(len(rows)))*scale
        mf[d] = csr_matrix((x, (rows, cols)), shape=h.intra.shape)
    mf[(0, 0, 0)] = (mf[(0, 0, 0)] + mf[(0, 0, 0)].conj().T).tocsr()
    return mf
