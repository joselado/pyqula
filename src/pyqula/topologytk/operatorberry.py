# Kubo-formula Berry curvature with an arbitrary operator, evaluated in the
# eigenbasis of the Hamiltonian at a single k-point (previously a compiled
# Fortran routine, ported here to numpy).
import numpy as np


def _berry_curvature_bands(dhdx, dhdy, waves, es, operator, delta):
    """Return the Berry curvature contribution of every band.

    waves[k,:] must be conj(psi_k), the same convention topology.py already
    builds via ws = np.conjugate(np.transpose(ws)) before calling this."""
    # Coerce to plain ndarrays first: multicell.derivative (current.py)
    # returns np.matrix whenever h.intra is one, and `*` on np.matrix is
    # MATRIX multiplication, not the elementwise product `prod` below needs.
    # Left unconverted this silently computed a matrix product there and
    # destroyed the Berry curvature (BZ integral came out ~0 instead of
    # 2*pi*Chern) -- see tests/topology/test_operator_berry_oracle.py.
    # NB: `operator` is deliberately NOT coerced -- callers such as
    # topology.spin_chern pass an Operator object, which implements @ itself;
    # np.asarray would collapse it to a 0-d object array. Coerce its product
    # with dhdx instead.
    dhdx = np.asarray(dhdx)
    dhdy = np.asarray(dhdy)
    waves = np.asarray(waves)
    es = np.asarray(es)
    opdhdx = np.asarray((operator@dhdx + dhdx@operator)/2.)
    h1 = waves@dhdy@np.conjugate(waves).T   # h1[jj,ii] = <jj|dhdy|ii>
    h2 = waves@opdhdx@np.conjugate(waves).T  # h2[ii,jj] = <ii|opdhdx|jj>
    prod = h1.T*h2 # prod[ii,jj] = <jj|dhdy|ii><ii|opdhdx|jj>
    denom = (es[:,None] - es[None,:])**2 + delta*delta
    np.fill_diagonal(denom,1.0) # avoid division by zero, masked out below
    contribution = np.imag(prod/denom)
    np.fill_diagonal(contribution,0.0) # ii==jj excluded
    # Overall MINUS sign. `contribution` above is
    # +Im[<n|dhdx|m><m|dhdy|n>]/(En-Em)^2; the factor 2 of the Kubo formula,
    # and the 2*pi per derivative that multicell.derivative omits (see
    # current.py:derivative), are together supplied by the 8*pi^2 that the
    # topology.py callers apply -- only the sign belongs here. Without it the
    # BZ integral came out with the opposite sign to every other Berry
    # quantity in the package.
    #
    # SIGN CONVENTION -- do not "correct" this to the textbook Kubo form.
    # What this sign buys is agreement with pyqula's OWN convention, not with
    # Xiao/Chang/Niu (RMP 82, 1959 (2010)). topology.berry_curvature returns
    # -Omega in the RMP convention (see its SIGN CONVENTION docstring: the
    # link-variable product is exp(-i*closed integral of A), so its argument
    # is minus the Berry phase), and every Chern number in the package
    # inherits that global sign. With the sign as written here,
    # operator_berry tracks berry_curvature pointwise (verified: ratio 1.00
    # across k on a gapped Haldane model) and their BZ integrals agree, which
    # is what makes topology.spin_chern and bandstructure.berry_bands
    # consistent with h.get_chern(). Flipping it to match the RMP formula
    # would put this one function out of step with the rest of pyqula.
    return -np.sum(contribution,axis=1) # sum over jj, one value per band


def berry_curvature_bands(dhdx, dhdy, waves, es, operator, delta):
    """Berry curvature of every band (occupied and empty)."""
    return _berry_curvature_bands(dhdx,dhdy,waves,es,operator,delta)


def berry_curvature(dhdx, dhdy, waves, es, operator, delta):
    """Total Berry curvature, summed over the occupied bands (es<=0)."""
    bouts = _berry_curvature_bands(dhdx,dhdy,waves,es,operator,delta)
    return np.sum(bouts[es<=0.])
