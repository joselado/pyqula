"""Generalized Stacking Fault Energy (GSFE) for the interlayer adhesion
between two graphene sheets.

This is the truncated Fourier expansion of the interlayer binding energy in
terms of the local interlayer registry (disregistry) vector, with the
graphene-specific coefficients fit in Carr, Massatt, Torrisi, Cazeaux,
Luskin, Kaxiras, "Relaxation and Domain Formation in Incommensurate 2D
Heterostructures", arXiv:1805.06972, Table 1 (their general form also has
two sine terms, which vanish identically for a graphene/graphene bilayer
because AB and BA stacking are degenerate). c0..c3 are in meV per graphene
unit cell -- since the periodic dependence on registry is entirely carried
by the dimensionless phases (v,w) below, this energy scale is independent
of whatever length units the geometry itself uses.
"""
import jax.numpy as jnp
import numpy as np

# Table 1 of arXiv:1805.06972, graphene/graphene column (meV per unit cell)
GRAPHENE_GSFE = dict(c0=6.832, c1=4.064, c2=-0.374, c3=-0.095)

# geometry.honeycomb_lattice()'s microscopic lattice vectors (bond
# length 1). The GSFE phase matrix must always be built from THESE, not
# from a supercell's own (possibly huge, moire-scale) a1,a2 -- b is the
# actual lab-frame registry shift between two graphene sheets, and its
# periodicity is set by the intrinsic monolayer lattice, unaffected by
# whatever simulation supercell happens to contain it.
MONOLAYER_A1 = np.array([1.5, np.sqrt(3.)/2])
MONOLAYER_A2 = np.array([-1.5, np.sqrt(3.)/2])


def reciprocal_vectors_2d(a1, a2):
    """Return (b1,b2), the 2D reciprocal vectors dual to real-space
    lattice vectors a1,a2 (ai.bj = 2*pi*delta_ij), taking only the
    in-plane (x,y) components."""
    a1 = np.asarray(a1)[:2]
    a2 = np.asarray(a2)[:2]
    area = a1[0]*a2[1] - a1[1]*a2[0]
    b1 = (2*np.pi/area)*np.array([a2[1], -a2[0]])
    b2 = (2*np.pi/area)*np.array([-a1[1], a1[0]])
    return b1, b2


def stacking_phases_matrix(a1=MONOLAYER_A1, a2=MONOLAYER_A2):
    """Return the 2x2 matrix M such that (v,w) = M @ b for an in-plane
    registry vector b, given the graphene layer's Bravais lattice vectors
    a1,a2 (honeycomb_lattice()'s convention: same length, ~120 degrees
    apart, one atom of each of the two sublattices per cell) -- defaults
    to MONOLAYER_A1,MONOLAYER_A2 above, the only physically correct choice
    for a geometry built out of honeycomb_lattice() cells (see its
    docstring); only override for testing.

    The second lattice vector is used with a flipped sign (-a2, an
    equally valid choice of primitive cell) -- this is the handedness
    that puts the AB stacking shift (a1-a2)/3 exactly at the GSFE minimum
    (v,w)=(2*pi/3,2*pi/3), and the inequivalent BA shift -(a1-a2)/3 at the
    degenerate minimum (4*pi/3,4*pi/3), matching the well-established fact
    that Bernal (AB/BA) stacking is graphene's energy minimum and AA its
    maximum. Checked directly against gsfe() below rather than assumed,
    since arXiv:1805.06972 states its (v,w) convention for a differently
    oriented (a1,a2) pair not reproduced here -- see
    tests/graphene/test_gsfe.py."""
    b1, b2 = reciprocal_vectors_2d(a1, -np.asarray(a2)[:2])
    return np.array([b1, b2])


def gsfe(v, w, c=GRAPHENE_GSFE):
    """Interlayer adhesion energy (meV per unit cell) at phases (v,w)."""
    c0, c1, c2, c3 = c["c0"], c["c1"], c["c2"], c["c3"]
    out = c0
    out = out + c1*(jnp.cos(v) + jnp.cos(w) + jnp.cos(v + w))
    out = out + c2*(jnp.cos(v + 2*w) + jnp.cos(v - w) + jnp.cos(2*v + w))
    out = out + c3*(jnp.cos(2*v) + jnp.cos(2*w) + jnp.cos(2*v + 2*w))
    return out


def gsfe_of_registry(b, phase_matrix, c=GRAPHENE_GSFE):
    """gsfe() evaluated directly on an in-plane registry vector b (...,2),
    using the (v,w)=phase_matrix@b convention of stacking_phases_matrix."""
    b = jnp.asarray(b)
    m = jnp.asarray(phase_matrix)
    v = m[0, 0]*b[..., 0] + m[0, 1]*b[..., 1]
    w = m[1, 0]*b[..., 0] + m[1, 1]*b[..., 1]
    return gsfe(v, w, c=c)
