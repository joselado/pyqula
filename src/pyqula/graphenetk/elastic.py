"""Intralayer elastic energy penalizing the relaxation displacement field
of a graphene sheet, following the linear isotropic elasticity functional
and graphene elastic constants (G,K, in meV per unit cell) of Carr,
Massatt, Torrisi, Cazeaux, Luskin, Kaxiras, "Relaxation and Domain
Formation in Incommensurate 2D Heterostructures", arXiv:1805.06972, Table
1: E = (1/2)[G*(dux/dx+duy/dy)^2 + K*((dux/dx-duy/dy)^2+(dux/dy+duy/dx)^2)]
per unit cell.

There is no continuous displacement-gradient field here, only one
relaxation displacement per atom, so each atom's local gradient is
estimated from its 3 nearest-neighbor bonds. Two more direct schemes were
tried and both turned out to be exploitable by the optimizer -- collapsing
real bond lengths at near-zero fitted energy cost, confirmed by
tests/graphene/test_relax.py's minimum bond-length check and by direct
single-atom probes (see git history for the numbers):

- A single least-squares fit of the 2x2 map F over all 3 bonds at once
  (d_n = F d0_n) is underdetermined in disguise: 3 bond observations
  supply 6 numbers for F's 4 unknowns, leaving 2 combinations of the 3
  bonds' individual motions that never affect the fitted F at all (their
  contribution is absorbed entirely into the fit's residual, which the
  energy never sees).
- Averaging the 3 *exact* pairwise fits (H_ij from bonds i,j alone, a
  bijective 4-equation/4-unknown system with zero residual) before
  computing the energy from H_avg does not fix this: forcing one bond to
  a given length still leaves 4 free numbers in the other two bonds to
  drive H_avg back to ~0 by cancellation between the three H_ij (3
  matrices summing to ~0 while individually large).

The fix is to average the *energies*, not the H's: e_atom = mean_ij
elastic_energy_density(H_ij). Each elastic_energy_density(H_ij) is a
positive semidefinite quadratic form (sum of squares), so summing three
of them admits no cancellation -- forcing e(H_ij)=0 for all three pairs
forces every bond to be an exact rigid rotation of its rigid counterpart,
i.e. no bond-length or bond-angle change at all. In the smooth/affine
limit all 3 H_ij coincide with the true displacement gradient H, so
mean_ij e(H_ij) reduces to e(H) exactly -- the G,K calibration against
Table 1 is unaffected."""
import jax.numpy as jnp

# Table 1 of arXiv:1805.06972, graphene column (meV per unit cell)
GRAPHENE_ELASTIC = dict(G=69518.0, K=47352.0)

_BOND_PAIRS = ((0, 1), (1, 2), (2, 0))


def pairwise_deformation_gradients(d0, d):
    """d0,d: (...,3,2) rigid/current in-plane bond vectors to the 3
    neighbors of each atom. Returns (...,3,2,2): the 3 exact pairwise
    displacement gradients H_ij[...,k,a,b] = du_a/dx_b, one per bond pair
    (0,1),(1,2),(2,0) (see module docstring)."""
    hs = []
    for i, j in _BOND_PAIRS:
        d0_cols = jnp.stack([d0[..., i, :], d0[..., j, :]], axis=-1)
        d_cols = jnp.stack([d[..., i, :], d[..., j, :]], axis=-1)
        f = d_cols @ jnp.linalg.inv(d0_cols)
        hs.append(f - jnp.eye(2))
    return jnp.stack(hs, axis=-3)


def elastic_energy_density(h, c=GRAPHENE_ELASTIC):
    """h: (...,2,2) displacement gradient, h[...,i,j]=du_i/dx_j. Returns
    the per-cell elastic energy (meV)."""
    g, k = c["G"], c["K"]
    dxux, dyux = h[..., 0, 0], h[..., 0, 1]
    dxuy, dyuy = h[..., 1, 0], h[..., 1, 1]
    dilation = dxux + dyuy
    shear1 = dxux - dyuy
    shear2 = dxuy + dyux
    return 0.5*(g*dilation**2 + k*(shear1**2 + shear2**2))


def cell_elastic_energy(d0, d, c=GRAPHENE_ELASTIC):
    """d0,d: (...,3,2) rigid/current in-plane bond vectors to the 3
    neighbors of each atom. Returns (...,), the per-cell elastic energy
    (meV): the mean over the 3 pairwise fits' *energies* (module
    docstring -- averaging the H's instead is exploitable)."""
    hs = pairwise_deformation_gradients(d0, d)
    return jnp.mean(elastic_energy_density(hs, c=c), axis=-1)
