"""Point-group symmetry detection and representation, used by
``wanniertk.wannierize``'s symmetry-enforced Wannierization
(``get_wannier_hamiltonian(..., symmetries=...)``).

Per pyqula's Bloch convention (``htk/bloch.py``: ``H(k) = intra +
sum_{R!=0} hopping[R] * exp(i*2*pi*R.k)``, with ``k`` fractional
reciprocal coordinates and ``R`` an integer lattice-vector index -- no
intracell atomic positions in the phase), a real-space point-group
operation ``Q`` about a Cartesian ``center`` that maps site ``i`` (home
cell) onto site ``pi(i)`` translated by integer lattice vector
``shift[i]`` induces, at every k:

    H(k') = P(k) @ H(k) @ P(k)^dagger,   k' = M^{-T} @ k

where ``M`` is ``Q``'s integer representation on the lattice vectors
(``Q @ a_j = sum_i M[i,j] * a_i``) and ``P(k)`` is block-sparse with
``P(k)[pi(i), i] = D_spin(Q) * exp(-i*2*pi* (k'.shift[i]))`` (derived in
the PR discussion for this module; a spinless Hamiltonian just drops
``D_spin``). This module never trusts that derivation blindly: every
operation, whether auto-detected or user-supplied, is re-verified
numerically against the actual ``h.get_hk_gen()`` before being reported
as (or used as) a real symmetry of a Hamiltonian -- see
:func:`compile_symmetry`. Geometry-only detection provides *candidates*;
Hamiltonian verification always narrows that set down (a Hamiltonian's
symmetries are a subset of its geometry's), never the reverse.

Scope: point-group operations only (no fractional/space-group
translations beyond the lattice-vector shift above, which is a bookkeeping
detail of embedding the operation into the Bloch representation, not a
genuine glide/screw symmetry), spinless or spinful (not Nambu/BdG --
``has_eh`` Hamiltonians are rejected, see ``wannierize.py``'s
integration). Orbital character beyond spin (e.g. p/d angular momentum)
is not modeled -- sites are treated as point-like/scalar per basis row,
matching ``wanniertk.wannierize``'s own orbital-basis convention.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from ..geometrytk.fractional import get_fractional_function

_PAULI = {
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "z": np.array([[1, 0], [0, -1]], dtype=complex),
}


class SymmetryOperation:
    """A single point-group operation: an orthogonal 3x3 Cartesian
    ``matrix`` (rotation or improper rotation/reflection), applied about
    a Cartesian ``center`` (default the origin). ``name`` is only used
    for messages/repr.

    This object alone does not know which sites/Hamiltonian it acts on --
    that mapping (site permutation, lattice-vector shifts, and whether
    the operation is actually a symmetry of a given ``Hamiltonian``) is
    always derived and verified fresh by :func:`compile_symmetry`."""

    def __init__(self, matrix, center=(0., 0., 0.), name=None):
        self.matrix = np.array(matrix, dtype=float)
        self.center = np.array(center, dtype=float)
        self.name = name or "R"
        if self.matrix.shape != (3, 3) or not np.allclose(
                self.matrix @ self.matrix.T, np.eye(3), atol=1e-6):
            raise ValueError(f"SymmetryOperation matrix must be a 3x3 orthogonal "
                              f"matrix, got {self.matrix}")

    def __repr__(self):
        return f"SymmetryOperation({self.name}, center={self.center.tolist()})"

    def spin_matrix(self):
        """(2,2) complex SU(2) matrix representing this operation's action
        on a spin-1/2 degree of freedom, built from the *proper*-rotation
        part only (``det(matrix)*matrix``) -- spin is a pseudovector and
        so is unaffected by the improper (mirror/inversion) part of the
        operation."""
        proper = np.linalg.det(self.matrix) * self.matrix
        rotvec = Rotation.from_matrix(proper).as_rotvec()
        angle = np.linalg.norm(rotvec)
        if angle < 1e-10:
            return np.eye(2, dtype=complex)
        axis = rotvec / angle
        nsigma = axis[0] * _PAULI["x"] + axis[1] * _PAULI["y"] + axis[2] * _PAULI["z"]
        return np.cos(angle / 2) * np.eye(2, dtype=complex) - 1j * np.sin(angle / 2) * nsigma


class CompiledSymmetry:
    """A :class:`SymmetryOperation` resolved against a specific geometry
    (site permutation ``perm``, integer lattice-vector ``shift`` per
    site) and, once :func:`compile_symmetry` has verified it, against a
    specific ``Hamiltonian`` -- everything :func:`orbital_operator` needs
    to build ``P(k)`` without repeating the geometry search."""

    def __init__(self, op, perm, shift, M, has_spin):
        self.op = op
        self.perm = np.asarray(perm, dtype=int)
        self.shift = np.asarray(shift, dtype=float)  # (n_sites, dim)
        self.M = M  # (dim,dim) int, or None if dim==0
        self.has_spin = has_spin
        self.spin_dim = 2 if has_spin else 1
        self._D_spin = op.spin_matrix() if has_spin else np.eye(1, dtype=complex)

    def k_image(self, k_frac):
        """``k' = M^{-T} @ k`` -- the fractional k-point this operation
        maps ``k_frac`` to."""
        dim = self.M.shape[0] if self.M is not None else 0
        if dim == 0:
            return np.array(k_frac)
        k = np.asarray(k_frac, dtype=float)[:dim]
        kp = np.linalg.inv(self.M).T @ k
        out = np.zeros(3)
        out[:dim] = kp
        return out

    def orbital_operator(self, k_frac):
        """``(P, k_image)``: the ``(num_orbitals,num_orbitals)`` unitary
        ``P(k)`` such that ``H(k_image) == P @ H(k_frac) @ P^dagger``
        for a Hamiltonian that actually has this symmetry, and the
        fractional k-point ``k_image`` it maps ``k_frac`` to."""
        kp = self.k_image(k_frac)
        dim = self.M.shape[0] if self.M is not None else 0
        n = len(self.perm)
        sd = self.spin_dim
        P = np.zeros((n * sd, n * sd), dtype=complex)
        for i in range(n):
            j = int(self.perm[i])
            phase = 1.0
            if dim > 0:
                phase = np.exp(-1j * 2 * np.pi * np.dot(kp[:dim], self.shift[i]))
            P[j * sd:(j + 1) * sd, i * sd:(i + 1) * sd] = self._D_spin * phase
        return P, kp


def _periodic_lattice_vectors(g):
    dim = g.dimensionality
    vecs = []
    if dim > 0: vecs.append(np.array(g.a1, dtype=float))
    if dim > 1: vecs.append(np.array(g.a2, dtype=float))
    if dim > 2: vecs.append(np.array(g.a3, dtype=float))
    return vecs


def _candidate_axes(g):
    axes = [np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.])]
    vecs = _periodic_lattice_vectors(g)
    for v in vecs:
        n = np.linalg.norm(v)
        if n > 1e-8: axes.append(v / n)
    if len(vecs) >= 2:
        normal = np.cross(vecs[0], vecs[1])
        n = np.linalg.norm(normal)
        if n > 1e-8: axes.append(normal / n)
    uniq = []
    for a in axes:
        if not any(np.allclose(a, u, atol=1e-6) or np.allclose(a, -u, atol=1e-6) for u in uniq):
            uniq.append(a)
    return uniq


def _mirror_matrix(normal):
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    return np.eye(3) - 2 * np.outer(n, n)


def _matrix_key(M):
    """Hashable key for a 3x3 matrix, rounded to avoid missing
    floating-point-noise duplicates. ``tuple(...tolist())``, not
    ``.tobytes()``: Python floats compare/hash ``-0.0 == 0.0``, so this
    (unlike raw byte comparison) isn't fooled by sign-of-zero differences
    between mathematically identical matrices."""
    return tuple(np.round(M, 6).flatten().tolist())


def _candidate_operations(axes, orders):
    """Candidate ``(name, matrix)`` pairs, deduplicated by matrix: e.g.
    ``S2`` about any axis is always exactly ``inversion``, and e.g.
    ``C4^2`` always exactly equals ``C2`` about the same axis (already
    generated separately whenever ``2`` is also in ``orders``, the
    default) -- generating and keeping such algebraic duplicates would
    only cost every downstream candidate consumer (``_match_geometry``,
    then ``compile_symmetry``'s Hamiltonian verification) repeated work
    for zero additional candidates. Deduplication never drops a
    *distinct* operation, regardless of which ``orders``/``axes`` are
    requested -- only bit-identical matrices are merged."""
    seen = set()
    ops = []

    def add(name, R):
        key = _matrix_key(R)
        if key in seen: return
        seen.add(key)
        ops.append((name, R))

    add("inversion", -np.eye(3))
    for axis in axes:
        axis_label = np.round(axis, 3).tolist()
        sigma_h = _mirror_matrix(axis)
        add(f"sigma(axis={axis_label})", sigma_h)
        for n in orders:
            for k in range(1, n):
                angle = 2 * np.pi * k / n
                R = Rotation.from_rotvec(axis * angle).as_matrix()
                label = f"C{n}" if k == 1 else f"C{n}^{k}"
                add(f"{label}(axis={axis_label})", R)
                S = sigma_h @ R
                slabel = f"S{n}" if k == 1 else f"S{n}^{k}"
                add(f"{slabel}(axis={axis_label})", S)
    return ops


def _lattice_matrix(vecs, Q, tol=1e-4):
    """Integer matrix ``M`` with ``Q @ a_j = sum_i M[i,j] * a_i``, or
    ``None`` if ``Q`` does not map the (periodic) lattice onto itself."""
    dim = len(vecs)
    if dim == 0:
        return np.zeros((0, 0), dtype=int)
    A = np.array(vecs).T  # (3,dim)
    QA = Q @ A
    M, _, _, _ = np.linalg.lstsq(A, QA, rcond=None)
    scale = max(1.0, float(np.max(np.abs(QA))))
    if np.max(np.abs(A @ M - QA)) > tol * scale:
        return None  # Q does not preserve the periodic subspace
    Mint = np.round(M)
    if np.max(np.abs(M - Mint)) > tol:
        return None  # not an integer (not a lattice symmetry)
    if abs(abs(np.linalg.det(Mint)) - 1) > tol:
        return None
    return Mint.astype(int)


def _match_geometry(g, op, vecs, tol_frac=1e-4, tol_pos=1e-4):
    """``(perm, shift, M)`` mapping every site to another site (up to an
    integer lattice-vector translation) under ``op``, or ``(None, None,
    None)`` if ``op`` is not a symmetry of ``g``."""
    M = _lattice_matrix(vecs, op.matrix, tol=tol_frac)
    if M is None:
        return None, None, None
    dim = g.dimensionality
    frac = get_fractional_function(g)
    r = np.array(g.r, dtype=float)
    n = len(r)
    perm = -np.ones(n, dtype=int)
    shift = np.zeros((n, dim))
    for i in range(n):
        ri = op.matrix @ (r[i] - op.center) + op.center
        fi = frac(ri)
        match = None
        for j in range(n):
            fj = np.array(g.frac_r[j])
            d = fi - fj
            if dim > 0:
                dper = d[:dim]
                rounded = np.round(dper)
                if np.max(np.abs(dper - rounded)) > tol_frac: continue
            else:
                rounded = np.zeros(0)
            dnon = d[dim:]
            if dnon.size and np.max(np.abs(dnon)) > tol_pos: continue
            match = (j, rounded)
            break
        if match is None:
            return None, None, None
        perm[i], shift[i] = match
    if len(set(perm.tolist())) != n:
        return None, None, None
    return perm, shift, M


def compile_symmetry(h, op, tol_frac=1e-4, tol_pos=1e-4, nk_check=6, op_tol=1e-6, rng=None,
                      hk_gen=None):
    """Resolve a :class:`SymmetryOperation` against ``h``'s geometry
    (site permutation/lattice shifts) and verify it against ``h`` itself
    by sampling random k-points and checking ``H(k') == P(k) H(k)
    P(k)^dagger`` (see module docstring). Returns a
    :class:`CompiledSymmetry`, or ``None`` if ``op`` is not a genuine
    symmetry of ``g``/``h`` -- callers should treat ``None`` as "this
    operation does not apply here", not an error, since geometry
    symmetries are only ever a superset of the Hamiltonian's own.

    ``hk_gen``, optional: a pre-built ``h.get_hk_gen()`` closure, reused
    as-is instead of rebuilding one from scratch -- ``get_hk_gen()`` redoes
    real work (multicell conversion, hopping filtering, dense-array setup),
    so callers that verify many candidate operations against the same ``h``
    in a loop (:func:`find_point_group`, :func:`close_group`) build it once
    and pass it through here instead of paying that cost per candidate."""
    if h.has_eh:
        raise NotImplementedError(
            "compile_symmetry: point-group symmetry enforcement is not implemented "
            "for Nambu/BdG Hamiltonians (h.has_eh=True)")
    g = h.geometry
    n_sites = len(g.r)
    num_orbitals = h.intra.shape[0]
    expected = 2 * n_sites if h.has_spin else n_sites
    if num_orbitals != expected:
        # same one-orbital-per-site(-and-spin) assumption
        # wanniertk.wannierize._orbital_positions_frac makes -- checked here
        # too so an unsupported (e.g. multi-orbital-per-site) Hamiltonian
        # gets this clear message instead of an opaque shape-mismatch deep
        # inside the P(k) verification below
        raise NotImplementedError(
            "compile_symmetry: point-group symmetry enforcement only supports one orbital "
            f"per site (times 2 for spin); got num_orbitals={num_orbitals} for {n_sites} "
            f"geometry sites (has_spin={h.has_spin})")
    if not getattr(g, "has_fractional", False): g.get_fractional()
    vecs = _periodic_lattice_vectors(g)
    perm, shift, M = _match_geometry(g, op, vecs, tol_frac=tol_frac, tol_pos=tol_pos)
    if perm is None:
        return None
    compiled = CompiledSymmetry(op, perm, shift, M, h.has_spin)
    if hk_gen is None:
        hk_gen = h.get_hk_gen()
    dim = h.dimensionality
    rng = rng or np.random.default_rng(0)
    for _ in range(nk_check):
        k = np.zeros(3); k[:dim] = rng.random(dim)
        P, kp = compiled.orbital_operator(k)
        Hk = np.asarray(hk_gen(k), dtype=complex)
        Hkp = np.asarray(hk_gen(kp), dtype=complex)
        if np.max(np.abs(Hkp - P @ Hk @ P.conj().T)) > op_tol:
            return None
    return compiled


def find_point_group(g, h=None, orders=(2, 3, 4, 6), axes=None, centers=None,
                      tol_frac=1e-4, tol_pos=1e-4, nk_check=6, op_tol=1e-6):
    """Best-effort, dependency-free search for point-group symmetries of
    geometry ``g`` (optionally narrowed to those verified as symmetries
    of Hamiltonian ``h`` too -- see :func:`compile_symmetry`).

    Candidates are built from ``orders`` (proper/improper rotation
    orders tried about each of ``axes``, default: the Cartesian x/y/z
    axes, each periodic lattice-vector direction, and -- for a 2D
    periodic ``g`` -- the layer normal) about each of ``centers``
    (default: the Cartesian origin only). This is a heuristic, not an
    exhaustive space-group search: symmetries centered elsewhere (e.g.
    a honeycomb lattice's hexagon centers, or any atom position when
    ``centers`` is left at its default) are not found automatically --
    pass ``centers=`` explicitly (e.g. including ``g.r``), or construct
    and pass a :class:`SymmetryOperation` directly, for those.

    Returns a list of :class:`SymmetryOperation` (if ``h`` is None) or
    :class:`CompiledSymmetry` (if ``h`` is given -- only operations that
    verify against ``h`` itself are returned, since a Hamiltonian's
    symmetries are always a subset of its geometry's)."""
    if not getattr(g, "has_fractional", False): g.get_fractional()
    if axes is None: axes = _candidate_axes(g)
    if centers is None: centers = [np.zeros(3)]
    vecs = _periodic_lattice_vectors(g)
    # built once and reused for every candidate below, not per-candidate --
    # see compile_symmetry's hk_gen docstring
    hk_gen = h.get_hk_gen() if h is not None else None
    candidates = _candidate_operations(axes, orders)  # already deduped, see its docstring
    found = []
    seen = set()
    for center in centers:
        for name, R in candidates:
            op = SymmetryOperation(R, center=center, name=name)
            perm, shift, M = _match_geometry(g, op, vecs, tol_frac=tol_frac, tol_pos=tol_pos)
            if perm is None: continue
            key = (_matrix_key(R), tuple(np.round(center, 6).tolist()), perm.tobytes())
            if key in seen: continue
            seen.add(key)
            if h is None:
                found.append(op)
            else:
                compiled = compile_symmetry(h, op, tol_frac=tol_frac, tol_pos=tol_pos,
                                             nk_check=nk_check, op_tol=op_tol, hk_gen=hk_gen)
                if compiled is not None: found.append(compiled)
    return found


def close_group(h, ops, max_order=48, tol_frac=1e-4, tol_pos=1e-4, nk_check=6, op_tol=1e-6):
    """Closure of ``ops`` (a list of :class:`SymmetryOperation` or
    :class:`CompiledSymmetry`, all sharing the same rotation ``center``)
    under matrix composition, including the identity, with every element
    (re-)verified against ``h`` via :func:`compile_symmetry`.

    Group-averaging (see ``wanniertk.wannierize``'s
    ``_enforce_point_group_symmetry``) needs a mathematically closed
    group, not just whatever finite set of generators was requested or
    auto-detected -- the exact-covariance argument for the averaged
    result only holds when summing over a set closed under composition
    (a composition of two verified symmetries of ``h`` is always itself
    a symmetry of ``h``, but it need not be one of the specific
    operations that were searched for/passed in). This is why this
    function exists as a separate, explicit step rather than trusting
    ``ops`` as-is.

    Restricted to operations sharing one rotation center: composing
    rotations about different centers generally requires a translation
    part (a screw/glide), which is outside this module's point-group-only
    scope (see module docstring).

    Any input already given as a :class:`CompiledSymmetry` (e.g. from
    :func:`find_point_group`) is trusted rather than re-verified from
    scratch -- it was already checked against this same ``h``, so
    re-running :func:`compile_symmetry` on it would only repeat work for
    the same answer. Newly-composed elements still always go through the
    full verification (composing two genuine symmetries of ``h`` is
    always itself one, but the composed *matrix* may not be one of the
    ones already checked)."""
    hk_gen = h.get_hk_gen()  # built once, reused by every compile_symmetry call below
    base_ops = [c.op if isinstance(c, CompiledSymmetry) else c for c in ops]
    precompiled = {_matrix_key(c.op.matrix): c for c in ops if isinstance(c, CompiledSymmetry)}
    if not base_ops:
        identity = SymmetryOperation(np.eye(3), name="E")
        return [compile_symmetry(h, identity, tol_frac=tol_frac, tol_pos=tol_pos,
                                  nk_check=nk_check, op_tol=op_tol, hk_gen=hk_gen)]
    center = base_ops[0].center
    for o in base_ops[1:]:
        if not np.allclose(o.center, center, atol=1e-6):
            raise ValueError(
                "close_group: all symmetries must share the same rotation center "
                f"(got {center.tolist()} and {o.center.tolist()}) -- composing operations "
                "about different centers needs a translation part, outside this module's "
                "point-group-only scope")

    identity = SymmetryOperation(np.eye(3), center=center, name="E")
    group = {_matrix_key(identity.matrix): identity}
    for o in base_ops:
        group.setdefault(_matrix_key(o.matrix), o)
    # worklist/BFS closure: only test products involving an element added
    # since the last round, instead of rescanning every already-checked
    # pair of the whole group on every pass (both a@b and b@a, since the
    # group need not be abelian)
    worklist = list(group.values())
    while worklist:
        current = list(group.values())
        next_worklist = []
        for a in worklist:
            for b in current:
                for m in (a.matrix @ b.matrix, b.matrix @ a.matrix):
                    k = _matrix_key(m)
                    if k in group: continue
                    if len(group) >= max_order:
                        raise ValueError(
                            f"close_group: the requested symmetries generate more than "
                            f"{max_order} elements -- this is almost certainly an unintended "
                            "continuous symmetry rather than a genuine finite point group; "
                            "pass a smaller/more specific set of symmetries")
                    new_op = SymmetryOperation(m, center=center, name=f"({a.name})*({b.name})")
                    group[k] = new_op
                    next_worklist.append(new_op)
        worklist = next_worklist
    compiled = []
    for k, o in group.items():
        c = precompiled.get(k)
        if c is None:
            c = compile_symmetry(h, o, tol_frac=tol_frac, tol_pos=tol_pos,
                                  nk_check=nk_check, op_tol=op_tol, hk_gen=hk_gen)
        if c is None:
            raise ValueError(
                f"close_group: {o.name}, generated by composing the requested symmetries, "
                "is not a verified symmetry of the Hamiltonian -- the requested symmetries "
                "do not close into a consistent group for this h")
        compiled.append(c)
    return compiled
