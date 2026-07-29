# Kubo-formula Berry curvature with an arbitrary operator, evaluated in the
# eigenbasis of the Hamiltonian at a single k-point (previously a compiled
# Fortran routine, ported here to numpy).
import numpy as np


def _berry_curvature_bands(dhdx, dhdy, waves, es, operator, delta):
    """Return the Berry curvature contribution of every band.

    waves[k,:] must be conj(psi_k), the same convention topology.py already
    builds via ws = np.conjugate(np.transpose(ws)) before calling this."""
    opdhdx = (operator@dhdx + dhdx@operator)/2.
    h1 = waves@dhdy@np.conjugate(waves).T   # h1[jj,ii] = <jj|dhdy|ii>
    h2 = waves@opdhdx@np.conjugate(waves).T  # h2[ii,jj] = <ii|opdhdx|jj>
    prod = h1.T*h2 # prod[ii,jj] = <jj|dhdy|ii><ii|opdhdx|jj>
    denom = (es[:,None] - es[None,:])**2 + delta*delta
    np.fill_diagonal(denom,1.0) # avoid division by zero, masked out below
    contribution = np.imag(prod/denom)
    np.fill_diagonal(contribution,0.0) # ii==jj excluded
    return np.sum(contribution,axis=1) # sum over jj, one value per band


def berry_curvature_bands(dhdx, dhdy, waves, es, operator, delta):
    """Berry curvature of every band (occupied and empty)."""
    return _berry_curvature_bands(dhdx,dhdy,waves,es,operator,delta)


def berry_curvature(dhdx, dhdy, waves, es, operator, delta):
    """Total Berry curvature, summed over the occupied bands (es<=0)."""
    bouts = _berry_curvature_bands(dhdx,dhdy,waves,es,operator,delta)
    return np.sum(bouts[es<=0.])
