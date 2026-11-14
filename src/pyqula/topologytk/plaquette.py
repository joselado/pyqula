"""Chern number of a group of states given on a mesh of a two-dimensional
torus, from the plaquette product of Fukui, Hatsugai and Suzuki
(cond-mat/0503172).

The engine is shared by every route that computes the Chern number of
something that is not a Bloch band of h: the collective modes of a mean
field (excitons and magnons, bsetk/topology.py), the null vectors of the
kernel of a response function (chitk/kernelchern.py), and, for its mesh
only, the winding of a Green's function (topologytk/greenwinding.py). So
what it publishes is the mesh and the fluxes of states already computed
on it, and the routes solve their own problem once per point.

The mesh is q[i,j] = (i/nq + s1, j/nq + s2, 0) in reduced coordinates,
mesh(nq, shift), and the plaquettes wrap around the torus with the states
of the first row and column. A shift keeps a special point out of the
mesh, for instance Q=0, where the acoustic magnon of a magnet sits at zero
energy. This is right only when the states at q+G span the same space as
the states at q, which is what a Bloch Hamiltonian of pyqula gives, since
it is periodic in the reduced momentum (the lattice gauge). An unshifted
mesh with nq a multiple of 6 holds Gamma, M and K of a hexagonal zone, and
one with nq even holds Gamma, X and M of a square one. These are the
points where a symmetry can force two modes to touch, and a touching that
falls on a point of the mesh is refused by the routes, while one between
two points is not seen. The default nq=12 holds all of them (measured on
the Kondo lattice of chitk/kernelchern.py with no Haldane term, whose two
spin exciton bands touch at K: nq=6 and 12 refuse it, nq=10 returns an
integer with no error).

The sign is the one of topology.berry_curvature and h.get_chern(): the
Berry connection is A = i<u|grad u> and the flux of a plaquette is minus
the argument of the product of its four link determinants.

What the integer means. The sum of the fluxes over 2 pi is an integer by
construction, whatever the mesh, since every link appears in two
plaquettes with opposite orientation. It is the Chern number of the bundle
only when the true Berry flux through every plaquette is below pi in
modulus: the flux is the argument of a product, so it is always returned
inside (-pi, pi], and a true flux of 1.2 pi comes back as -0.8 pi, which
changes the integer with no other sign. The largest returned flux,
ModeChern.max_flux, is therefore the thing to look at: close to pi means
that nq has to grow. A group of n states that share one vector, as the
three components of a spin triplet do, has n times the flux of a single
state, so it needs a finer mesh. No fraction of pi is a sharp test, so
nothing here raises on the flux.

The other way the integer fails is a group that is not separated by a gap
from the rest, or a mesh so coarse that the states change by too much
between two neighbours; the overlap of the two subspaces on some link then
goes to zero. So the smallest normalized link overlap is returned with the
fluxes, and chern_from_states raises below a tolerance instead of
returning an integer that means nothing.

A metric, when given, is the inner product <a|S|b> used for the links. For
a positive metric the Chern number does not depend on it, since it is a
property of the bundle the states span and not of the inner product
(tests/topology/test_plaquette_chern.py checks it); the Berry curvature of
a single plaquette does depend on it. The same argument says that a
positive-norm collective mode of a Casida (bosonic Bogoliubov) problem
gives the same integer with metric=None and with its own indefinite metric
diag(1,-1), since that metric is positive on the mode (measured on two
models in bsetk/topology.py). With an indefinite metric the normalized
overlap is no longer bounded by one."""

import numpy as np

from .. import parallel


# the default number of points of the mesh along each direction, the same
# in every routine of the routes (see the module docstring)
NQ = 12


def mesh(nq=NQ, shift=(0., 0.)):
    """Return the points of the nq x nq mesh of the reduced Brillouin zone
    as an (nq, nq, 3) array, q[i,j] = (i/nq + s1, j/nq + s2, 0).

    nq: number of points along each reciprocal lattice vector, an integer
        of at least 2.
    shift: (s1, s2), the offset of the mesh in reduced coordinates."""
    if int(nq) != nq or nq < 2:
        raise ValueError("nq is the number of points of the mesh along each "
                "direction and must be an integer of at least 2, got "
                +str(nq))
    nq = int(nq)
    shift = np.array(shift, dtype=np.float64).reshape(-1)
    if shift.shape[0] != 2:
        raise ValueError("shift is the offset of the mesh in reduced "
                "coordinates and must have two components, got "+str(shift))
    q = np.zeros((nq, nq, 3))
    i = np.arange(nq)/nq
    q[:, :, 0] = i[:, None] + shift[0]
    q[:, :, 1] = i[None, :] + shift[1]
    return q


def require_2d(h, what):
    """Raise unless h is two-dimensional. what is the noun phrase the
    message names, e.g. "a magnon band"."""
    if h.dimensionality != 2:
        raise ValueError("the Chern number of %s needs a two-dimensional "
            "Hamiltonian, got dimensionality %d"%(what, h.dimensionality))


def check_bands(bands):
    """The group of modes as a sorted list of indices counted from the
    lowest mode, contiguous, so a degenerate multiplet goes in as one
    group and the gaps to the rest are well defined"""
    if np.ndim(bands) == 0: bands = [bands]
    out = []
    for b in bands:
        if int(b) != b or b < 0:
            raise ValueError("bands are indices of the modes counted from "
                "the lowest one, non-negative integers, got %r"%(b,))
        out.append(int(b))
    out = sorted(set(out))
    if len(out) == 0:
        raise ValueError("bands is empty; give the indices of the group of "
            "modes, e.g. [0] for the lowest one")
    if out != list(range(out[0], out[-1]+1)):
        raise ValueError("bands must be a contiguous group of modes, got %s. "
            "A mode left out of the middle of a group is a group of its own, "
            "with a Chern number of its own"%(out,))
    return out


def _as_states(v, where):
    """The states at one point as a complex (n, D) array, with n >= 1"""
    v = np.array(v, dtype=np.complex128)
    if v.ndim == 1: v = v[None, :] # a single state
    if v.ndim != 2:
        raise ValueError("the states must be a (n, D) array, one state per "
                "row, got shape "+str(v.shape)+" at "+where)
    if v.shape[0] == 0:
        raise ValueError("there is no state at "+where)
    return v


def states_on_mesh(states, nq=NQ, shift=(0., 0.)):
    """Evaluate states(q) on the points of mesh(nq, shift), once per point,
    and return a list of lists V[i][j], the states at mesh[i,j]."""
    if not callable(states):
        raise TypeError("states must be a callable q -> (n, D) array, got "
                +str(type(states)))
    qs = mesh(nq, shift)
    nq = qs.shape[0]
    vs = parallel.pcall(states, list(qs.reshape(-1, 3))) # one solve per point
    return [[vs[i*nq + j] for j in range(nq)] for i in range(nq)]


def _link(va, vb, metric):
    """Determinant of the overlap matrix <va_i|S|vb_j> and its modulus
    normalized by the Gram determinants of the two groups"""
    if metric is None: Svb = vb
    else: Svb = vb*metric[None, :]
    m = np.conj(va)@Svb.T
    d = np.linalg.det(m)
    ga = np.linalg.det(np.conj(va)@(va if metric is None else va*metric[None, :]).T)
    gb = np.linalg.det(np.conj(vb)@Svb.T)
    norm = np.sqrt(np.abs(ga*gb))
    return d, (np.abs(d)/norm if norm > 0. else 0.)


def _check_metric(metric, dim):
    if metric is None: return None
    metric = np.array(metric, dtype=np.float64).reshape(-1)
    if metric.shape[0] != dim:
        raise ValueError("the metric must be the diagonal of the inner "
                "product, one real number per component of a state ("
                +str(dim)+"), got "+str(metric.shape[0]))
    return metric


def mesh_berry_fluxes(V, metric=None):
    """Return (fluxes, min_overlap) from states already computed on a mesh:
    the Berry flux of each plaquette as an (nq, nq) array, whose sum over
    2 pi is the Chern number, and the smallest normalized overlap of two
    neighbouring groups of states over every link (1 for identical
    subspaces, 0 for orthogonal ones).

    V: V[i][j] is the (n, D) array of the n states at the point mesh[i,j]
        of an nq x nq mesh, one state per row, in a space that does not
        depend on q. Their normalization and the basis chosen inside the
        group do not matter, and n and D must be the same at every point.
    metric: None for the Euclidean inner product, or the diagonal of the
        inner product <a|S|b> as one real number per component.

    See the module docstring for when the integer is the Chern number."""
    nq = len(V)
    if nq < 2 or any(len(row) != nq for row in V):
        raise ValueError("the states must be given on an nq x nq mesh with "
                "nq of at least 2, as V[i][j]; got "+str(nq)+" rows of "
                "lengths "+str([len(row) for row in V]))
    V = [[_as_states(V[i][j], "the point (%d, %d) of the mesh"%(i, j))
          for j in range(nq)] for i in range(nq)]
    shape = V[0][0].shape
    for i in range(nq):
        for j in range(nq):
            if V[i][j].shape != shape:
                raise ValueError("the number of states (or their dimension) "
                    "changes over the mesh: "+str(shape)+" at the point "
                    "(0, 0) but "+str(V[i][j].shape)+" at the point "
                    +str((i, j))+". A group of states has a Chern number "
                    "only when it is separated by a gap from the rest at "
                    "every q, so it has the same size everywhere")
    metric = _check_metric(metric, shape[1])
    lx = np.zeros((nq, nq), dtype=np.complex128) # link (i,j) -> (i+1,j)
    ly = np.zeros((nq, nq), dtype=np.complex128) # link (i,j) -> (i,j+1)
    worst = 1.
    for i in range(nq):
        for j in range(nq):
            d, o = _link(V[i][j], V[(i+1) % nq][j], metric)
            lx[i, j] = d ; worst = min(worst, o)
            d, o = _link(V[i][j], V[i][(j+1) % nq], metric)
            ly[i, j] = d ; worst = min(worst, o)
    # plaquette (i,j),(i+1,j),(i+1,j+1),(i,j+1), counterclockwise, as the
    # corners of topology.berry_curvature
    p = lx*np.roll(ly, -1, axis=0)*np.conj(np.roll(lx, -1, axis=1))*np.conj(ly)
    fluxes = -np.angle(p)
    return fluxes, worst


def berry_fluxes(states, nq=NQ, metric=None, shift=(0., 0.)):
    """Return (fluxes, min_overlap) of the group of states returned by the
    callable states(q) on the points of mesh(nq, shift), as
    mesh_berry_fluxes returns them.

    states: callable q -> (n, D) array, the n states at the reduced
        momentum q = [q1, q2, 0] as rows, in a space that does not depend
        on q. Their normalization and the basis chosen inside the group
        do not matter.
    nq: number of points of the mesh along each reciprocal lattice vector.
    metric: None for the Euclidean inner product, or the diagonal of the
        inner product <a|S|b> as one real number per component.
    shift: offset of the mesh in reduced coordinates, two numbers."""
    return mesh_berry_fluxes(states_on_mesh(states, nq=nq, shift=shift),
                             metric=metric)


def check_overlap(worst, tol, nq, what="groups of states", detail=""):
    """Raise when the smallest link overlap is below tol"""
    if worst < tol:
        raise ValueError("two neighbouring %s of the mesh are almost "
            "orthogonal (smallest normalized overlap %.2e, below tol=%.1e), "
            "so the plaquette sum does not resolve their Chern number. "
            "Either the group is not separated by a gap from the rest at "
            "some q%s, or nq=%d is too coarse; raise nq to tell the two "
            "apart" % (what, worst, tol, detail, nq))


def chern_from_states(states, nq=NQ, metric=None, tol=1e-6, shift=(0., 0.)):
    """Chern number of the group of states returned by states(q), from the
    plaquette product on the nq x nq mesh of the reduced Brillouin zone.

    states, nq, metric and shift are those of berry_fluxes. tol is the
    smallest normalized link overlap accepted: below it the mesh does not
    resolve the states, or the group is not separated by a gap from the
    rest, and the integer means nothing, so a ValueError is raised instead.
    Returns a float that is an integer up to roundoff; see the module
    docstring for when it is the Chern number."""
    fluxes, worst = berry_fluxes(states, nq=nq, metric=metric, shift=shift)
    check_overlap(worst, tol, len(fluxes))
    return float(np.sum(fluxes)/(2.*np.pi))


class ModeChern():
    """The Chern number of a group of collective modes on a mesh of q, with
    what it rests on. It is what full_output=True of the routes returns
    (bsetk.topology.mode_chern, chitk.kernelchern.kernel_chern):

      chern       the plaquette sum over 2 pi, an integer up to roundoff
      fluxes      (nq,nq) Berry flux of each plaquette, inside (-pi, pi]
      max_flux    the largest |flux| of a plaquette; the integer is the
                  Chern number only when every true flux is below pi, so a
                  value close to pi means that nq has to grow (a group of n
                  modes that share one vector has n times the flux of one)
      min_overlap smallest normalized overlap of two neighbouring groups
      qs          (nq,nq,3) the mesh points, reduced coordinates
      energies    (nq,nq,n) energies of the n modes of the group at qs
      gap         smallest distance, over the mesh, of the group to the
                  other modes (route 1: the other modes of the excitation
                  branch; route 2: the other zeros of the window, or an end
                  of the window when there is none in between)
      bound       smallest distance, over the mesh, of the top of the group
                  below the continuum (route 1: the lowest electron-hole
                  or Stoner transition of the pair basis; route 2: the
                  upper edge of the continuum of the kernel), negative when
                  the group is inside it somewhere; None for a kernel with
                  no continuum

    gap and bound are taken on the points of the mesh only: a touching
    between two points is not seen, and a second nq is the check."""
    def __init__(self, fluxes, min_overlap, qs, energies, gap, bound):
        self.fluxes = np.array(fluxes, dtype=np.float64)
        nq = self.fluxes.shape[0]
        self.chern = float(np.sum(self.fluxes)/(2.*np.pi))
        self.max_flux = float(np.max(np.abs(self.fluxes)))
        self.min_overlap = min_overlap
        self.qs = np.array(qs, dtype=np.float64).reshape(nq, nq, 3)
        self.energies = np.array(energies, dtype=np.float64).reshape(nq, nq,
                                                                     -1)
        self.gap = float(gap)
        self.bound = None if bound is None else float(bound)
