"""Structural relaxation of a graphene multilayer by minimizing the sum of
the GSFE interlayer adhesion energy (gsfe.py) and the intralayer elastic
energy (elastic.py) over an in-plane relaxation displacement field, one
vector per atom.

Both energy terms are tabulated in meV per graphene unit cell (Table 1 of
arXiv:1805.06972 -- see gsfe.py/elastic.py), so each term is evaluated
using one representative atom per unit cell rather than both sublattice
atoms, which would double count: for the elastic term, that representative
is the sublattice=+1 atom carrying the local 3-neighbor deformation
gradient of its own cell; for the interlayer term, it is the same
sublattice=+1 atom matched to its nearest sublattice=+1 partner in the
neighboring layer.

Bond/partner topology (which periodic image of which neighbor atom) is
fixed once from the RIGID, unrelaxed geometry and reused throughout the
minimization -- valid as long as the relaxation displacement stays a small
fraction of the lattice constant, which is the regime this phenomenological
potential is meant to describe (it has no mechanism for bond breaking/
reconnection anyway).
"""
import numpy as np
import jax
import jax.numpy as jnp
from scipy.spatial import cKDTree
from scipy.optimize import minimize

from . import gsfe as gsfetk
from . import elastic as elastictk

jax.config.update("jax_enable_x64", True)


def _layer_groups(z, tol=1e-3):
    """Group atom indices by z-coordinate, ordered from lowest to highest
    z (each flat, unrelaxed graphene sheet has one z value per layer)."""
    order = np.argsort(z)
    zs = z[order]
    groups, current = [], [order[0]]
    for idx, zval in zip(order[1:], zs[1:]):
        if zval - z[current[-1]] > tol:
            groups.append(np.array(current))
            current = [idx]
        else:
            current.append(idx)
    groups.append(np.array(current))
    return groups


def _periodic_replicas(r2d, a1_2d, a2_2d, nrep=1):
    """Replica positions (K*N,2), origin atom index (K*N,) and applied
    real-space offset (K*N,2) for images in [-nrep,nrep]x[-nrep,nrep]
    unit cells of (a1_2d,a2_2d)."""
    ns = np.arange(-nrep, nrep + 1)
    offsets = np.array([n*a1_2d + m*a2_2d for n in ns for m in ns])
    n = len(r2d)
    rep_r = (r2d[None, :, :] + offsets[:, None, :]).reshape(-1, 2)
    rep_origin = np.tile(np.arange(n), len(offsets))
    rep_offset = np.repeat(offsets, n, axis=0)
    return rep_r, rep_origin, rep_offset


def _intralayer_cells(r2d, a1_2d, a2_2d, layer_idx, k=3, nrep=1):
    """One elastic 'cell' per atom of layer_idx (both sublattices -- see
    energy_function's 0.5 weight for why this does not double count): its
    k nearest neighbors within the same layer (necessarily the opposite
    sublattice -- the 3 bonded partners of a honeycomb site). Returns
    (owner, nbr, d0) with owner,nbr full-geometry indices (owner repeated
    k times) and d0 (ncells,k,2) the rigid in-plane bond vectors."""
    rep_r, rep_origin, rep_offset = _periodic_replicas(r2d[layer_idx], a1_2d, a2_2d, nrep=nrep)
    tree = cKDTree(rep_r)
    dist, idx = tree.query(r2d[layer_idx], k=k + 1)
    owner, nbr, d0 = [], [], []
    for a_local in range(len(layer_idx)):
        found = 0
        for cand in idx[a_local]:
            if rep_origin[cand] == a_local and np.allclose(rep_offset[cand], 0):
                continue
            owner.append(layer_idx[a_local])
            nbr.append(layer_idx[rep_origin[cand]])
            d0.append(rep_r[cand] - r2d[layer_idx][a_local])
            found += 1
            if found == k:
                break
    owner = np.array(owner).reshape(-1, k)[:, 0]
    nbr = np.array(nbr).reshape(-1, k)
    d0 = np.array(d0).reshape(-1, k, 2)
    return owner, nbr, d0


def _interlayer_cells(r2d, a1_2d, a2_2d, layer_from, layer_to, sublattice, nrep=1):
    """One GSFE 'cell' per atom of layer_from (both sublattices -- see
    energy_function's 0.5 weight), matched to its nearest SAME-sublattice
    partner in layer_to (the registry vector's zero is AA stacking, where
    like sublattices sit exactly on top of each other). Returns (owner,
    nbr, b0) with owner,nbr full-geometry indices and b0 (ncells,2) the
    rigid in-plane registry vector."""
    owner_l, nbr_l, b0_l = [], [], []
    for s in (1.0, -1.0):
        from_local = np.where(sublattice[layer_from] == s)[0]
        to_local = np.where(sublattice[layer_to] == s)[0]
        rep_r, rep_origin, _ = _periodic_replicas(r2d[layer_to][to_local], a1_2d, a2_2d, nrep=nrep)
        tree = cKDTree(rep_r)
        _, idx = tree.query(r2d[layer_from][from_local], k=1)
        owner_l.append(layer_from[from_local])
        nbr_l.append(layer_to[to_local[rep_origin[idx]]])
        b0_l.append(rep_r[idx] - r2d[layer_from][from_local])
    return np.concatenate(owner_l), np.concatenate(nbr_l), np.concatenate(b0_l)


def build_topology(g, nrep=1, layer_pairs=None):
    """Precompute, from the RIGID (unrelaxed) geometry `g`, everything
    needed by energy_function(): the fixed intralayer/interlayer cell
    bond topology, and the GSFE phase matrix. Returns a dict; see
    relax_structure for the `g`/`layer_pairs` requirements."""
    if not getattr(g, "has_sublattice", False):
        raise ValueError("this needs a geometry with has_sublattice=True "
                          "(e.g. built from geometry.honeycomb_lattice())")
    r = np.array(g.r, dtype=float)
    r2d = r[:, :2]
    sublattice = np.array(g.sublattice, dtype=float)
    a1_2d = np.array(g.a1[:2], dtype=float)
    a2_2d = np.array(g.a2[:2], dtype=float)
    layers = _layer_groups(r[:, 2])
    if layer_pairs is None:
        layer_pairs = [(k, k + 1) for k in range(len(layers) - 1)]
    d_nn, _ = cKDTree(r2d[layers[0]]).query(r2d[layers[0]], k=2)
    if abs(np.median(d_nn[:, 1]) - 1.0) > 1e-3:
        raise ValueError("relax_structure assumes geometry.honeycomb_lattice()'s "
                          "bond length (1.0); got a nearest-neighbor distance of "
                          f"{np.median(d_nn[:, 1]):.4f} in layer 0 -- the GSFE/elastic "
                          "coefficients from arXiv:1805.06972 do not apply as-is to a "
                          "rescaled lattice")

    owner_intra, nbr_intra, d0_intra = zip(*[
        _intralayer_cells(r2d, a1_2d, a2_2d, layer_idx, nrep=nrep)
        for layer_idx in layers])
    owner_inter, nbr_inter, b0_inter = zip(*[
        _interlayer_cells(r2d, a1_2d, a2_2d, layers[kl], layers[ku], sublattice, nrep=nrep)
        for kl, ku in layer_pairs])

    layer_id = np.zeros(len(r2d), dtype=np.int64)
    for li, layer_idx in enumerate(layers):
        layer_id[layer_idx] = li
    layer_counts = np.array([len(layer_idx) for layer_idx in layers], dtype=float)

    return dict(
        natoms=len(r2d),
        nlayers=len(layers),
        layer_id=jnp.array(layer_id),
        layer_counts=jnp.array(layer_counts),
        owner_intra=jnp.array(np.concatenate(owner_intra)),
        nbr_intra=jnp.array(np.concatenate(nbr_intra)),
        d0_intra=jnp.array(np.concatenate(d0_intra)),
        owner_inter=jnp.array(np.concatenate(owner_inter)),
        nbr_inter=jnp.array(np.concatenate(nbr_inter)),
        b0_inter=jnp.array(np.concatenate(b0_inter)),
        phase_matrix=jnp.array(gsfetk.stacking_phases_matrix()),
    )


def energy_function(topo, gsfe_coeffs=gsfetk.GRAPHENE_GSFE,
                     elastic_coeffs=elastictk.GRAPHENE_ELASTIC):
    """Return a jax-differentiable `energy(u_flat)` (u_flat: flattened
    (natoms,2) in-plane displacement -> total energy in meV) built from a
    build_topology() dict.

    Both sublattices independently contribute a cell term (see
    _intralayer_cells/_interlayer_cells) -- each weighted by 1/2 since two
    atoms make up one unit cell of the meV-per-unit-cell tabulated
    coefficients (gsfe.py/elastic.py). Constraining both sublattices
    directly, rather than only one representative per cell, matters here:
    a single-representative scheme leaves the other sublattice's
    displacement only indirectly (and too weakly) constrained through its
    neighbors' independent least-squares strain fits, which in practice
    let bonds collapse to a fraction of their rigid length at no
    appreciable energy cost -- caught by tests/graphene/test_relax.py's
    minimum-bond-length check.

    Each layer's mean displacement is subtracted out before evaluating
    the energy: a uniform shift of one whole layer relative to another
    costs zero intralayer elastic energy (it changes no bond within
    either layer) and, once a moire cell is large enough to sample
    registry space densely, next to nothing in interlayer GSFE either
    (shifting every cell's registry by the same vector barely moves
    Sum_i GSFE(b_i+c)) -- an almost-flat direction that otherwise
    dominates the raw displacement (empirically ~0.47 of a bond length,
    vs ~0.01 for the actual local/domain-forming relaxation, at a 2
    degree twist) and needlessly slows convergence. Gauge-fixing it here,
    rather than projecting it out of the result afterwards, makes it
    exactly unobservable to the optimizer instead of merely flat."""
    natoms = topo["natoms"]
    layer_id, layer_counts = topo["layer_id"], topo["layer_counts"]
    owner_intra, nbr_intra, d0_intra = topo["owner_intra"], topo["nbr_intra"], topo["d0_intra"]
    owner_inter, nbr_inter, b0_inter = topo["owner_inter"], topo["nbr_inter"], topo["b0_inter"]
    phase_matrix = topo["phase_matrix"]

    def energy(u_flat):
        u = u_flat.reshape(natoms, 2)
        layer_mean = jnp.zeros((topo["nlayers"], 2)).at[layer_id].add(u)/layer_counts[:, None]
        u = u - layer_mean[layer_id]
        d = d0_intra + (u[nbr_intra] - u[owner_intra][:, None, :])
        e_cell = elastictk.cell_elastic_energy(d0_intra, d, c=elastic_coeffs)
        e_intra = 0.5*jnp.sum(e_cell)
        b = b0_inter + (u[nbr_inter] - u[owner_inter])
        e_inter = 0.5*jnp.sum(gsfetk.gsfe_of_registry(b, phase_matrix, c=gsfe_coeffs))
        return e_intra + e_inter

    return energy


def relax_structure(g, nrep=1, maxiter=500, verbose=False, layer_pairs=None,
                     gsfe_coeffs=gsfetk.GRAPHENE_GSFE,
                     elastic_coeffs=elastictk.GRAPHENE_ELASTIC):
    """Relax a graphene multilayer geometry `g` (periodic, has_sublattice)
    by minimizing GSFE(interlayer) + elastic(intralayer) over an in-plane
    displacement field, one 2D vector per atom (jax autodiff gradient
    passed to scipy L-BFGS-B). Returns a new geometry of the same type as
    `g`, in-plane positions displaced; z is untouched (in-plane relaxation
    only -- no corrugation).

    `layer_pairs`: which (layer_index_lower, layer_index_upper) pairs get
    an interlayer GSFE term, indexing layers ordered by z; defaults to all
    adjacent pairs (nearest-layer adhesion only, the standard
    approximation -- see the module docstring)."""
    topo = build_topology(g, nrep=nrep, layer_pairs=layer_pairs)
    natoms = topo["natoms"]
    energy = energy_function(topo, gsfe_coeffs=gsfe_coeffs, elastic_coeffs=elastic_coeffs)
    value_and_grad = jax.jit(jax.value_and_grad(energy))

    def objective(x):
        v, dv = value_and_grad(jnp.array(x))
        return float(v), np.array(dv, dtype=np.float64)

    x0 = np.zeros(2*natoms)
    e0, g0 = objective(x0)
    # G,K ~ 1e4-1e5 meV make the default L-BFGS-B ftol/gtol (relative to a
    # machine-epsilon-scaled function value) stop far too early -- tightened
    # explicitly rather than relying on the scale-dependent defaults.
    res = minimize(objective, x0, jac=True, method="L-BFGS-B",
                    options=dict(maxiter=maxiter, ftol=1e-12, gtol=1e-8))
    if verbose:
        gnorm = np.max(np.abs(res.jac))
        print(f"relax_structure: E0={e0:.4f} meV (|grad|inf={np.max(np.abs(g0)):.2e}), "
              f"E_relaxed={res.fun:.4f} meV, converged={res.success}, "
              f"iters={res.nit}, |grad|inf={gnorm:.2e}, message={res.message}")

    u = res.x.reshape(natoms, 2)
    layer_id = np.array(topo["layer_id"])
    for li in range(topo["nlayers"]):
        u[layer_id == li] -= u[layer_id == li].mean(axis=0)
    g2 = g.copy()
    r2 = np.array(g2.r, dtype=float)
    r2[:, 0] += u[:, 0]
    r2[:, 1] += u[:, 1]
    g2.r = r2
    g2.x, g2.y, g2.z = r2[:, 0], r2[:, 1], r2[:, 2]
    return g2
