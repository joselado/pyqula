# Kubo-formula Berry curvature with an arbitrary operator, evaluated in the
# eigenbasis of the Hamiltonian at a single k-point (previously a compiled
# Fortran routine, ported here to numpy).
import numpy as np
from .. import algebra


def _todense(m):
    """Coerce a Hamiltonian derivative or an operator to a plain dense
    ndarray. Three shapes reach this module and none of them is one:

    - np.matrix, which multicell.derivative (current.py) returns whenever
      h.intra is one. `*` on np.matrix is MATRIX multiplication, not the
      elementwise product `prod` below needs; left unconverted this
      silently computed a matrix product and destroyed the Berry curvature
      (BZ integral came out ~0 instead of 2*pi*Chern) -- see
      tests/topology/test_operator_berry_oracle.py.
    - a scipy sparse matrix, which multicell.derivative returns whenever
      h.is_sparse -- which multicell.supercell_hamiltonian sets
      unconditionally, so every h.get_supercell(...) lands here.
    - an operators.Operator, the form h.get_operator(...) returns and the
      canonical way to name an operator in this library.

    np.asarray handles only the first: it wraps a sparse matrix (and an
    Operator) in a 0-d OBJECT array, and the next matmul then fails with an
    unreadable shape error. algebra.todense handles np.matrix and sparse
    alike, and an Operator is unwrapped to its matrix first."""
    if hasattr(m,"get_matrix"): # operators.Operator
        om = m.get_matrix() # None if it is defined only by its action
        if om is None:
            raise NotImplementedError("the operator-resolved Berry curvature "
                    "needs an operator with a matrix representation, and this "
                    "Operator is defined only by its action on a "
                    "wavefunction; build it from a matrix instead")
        m = om
    return algebra.todense(m)


def _berry_curvature_bands(dhdx, dhdy, waves, es, operator, delta):
    """Return the Berry curvature contribution of every band.

    waves[k,:] must be conj(psi_k), the same convention topology.py already
    builds via ws = np.conjugate(np.transpose(ws)) before calling this."""
    # Coerce everything to plain dense ndarrays first -- see _todense for
    # the three input shapes that are not, and what each of them breaks.
    dhdx = _todense(dhdx)
    dhdy = _todense(dhdy)
    waves = np.asarray(waves)
    es = np.asarray(es)
    operator = _todense(operator)
    opdhdx = (operator@dhdx + dhdx@operator)/2.
    h1 = waves@dhdy@np.conjugate(waves).T   # h1[jj,ii] = <jj|dhdy|ii>
    h2 = waves@opdhdx@np.conjugate(waves).T  # h2[ii,jj] = <ii|opdhdx|jj>
    prod = h1.T*h2 # prod[ii,jj] = <jj|dhdy|ii><ii|opdhdx|jj>
    denom = (es[:,None] - es[None,:])**2 + delta*delta
    np.fill_diagonal(denom,1.0) # avoid division by zero, masked out below
    contribution = np.imag(prod/denom)
    np.fill_diagonal(contribution,0.0) # ii==jj excluded
    # Overall MINUS sign -- it is the textbook Kubo sign, not a pyqula
    # convention. `contribution` above is
    # +Im[<n|dhdx|m><m|dhdy|n>]/(En-Em)^2, while the Berry curvature of
    # Xiao/Chang/Niu (RMP 82, 1959 (2010), A = i<u|grad_k u>) is
    # Omega_n = -2 Im sum_m <n|dH/dkx|m><m|dH/dky|n>/(En-Em)^2. The factor 2
    # of that formula, and the 2*pi per derivative that multicell.derivative
    # omits (see current.py:derivative), are together supplied by the 8*pi^2
    # that the topology.py callers apply -- only the sign belongs here.
    # Without it the BZ integral came out with the opposite sign to every
    # other Berry quantity in the package.
    #
    # topology.berry_curvature returns +Omega in that same RMP convention
    # (see its SIGN CONVENTION docstring), so with the sign as written here
    # operator_berry tracks it pointwise, which is what makes
    # topology.spin_chern and bandstructure.berry_bands consistent with
    # h.get_chern(). tests/topology/test_operator_berry_spin_blocks.py pins
    # the sign AND the absolute normalization against the Wilson-loop
    # curvature of each spin block of a Kane-Mele Hamiltonian.
    return -np.sum(contribution,axis=1) # sum over jj, one value per band


def berry_curvature_bands(dhdx, dhdy, waves, es, operator, delta):
    """Berry curvature of every band (occupied and empty)."""
    return _berry_curvature_bands(dhdx,dhdy,waves,es,operator,delta)


def berry_curvature(dhdx, dhdy, waves, es, operator, delta):
    """Total Berry curvature, summed over the occupied bands (es<=0)."""
    bouts = _berry_curvature_bands(dhdx,dhdy,waves,es,operator,delta)
    return np.sum(bouts[es<=0.])
