# routines to compute the stacking in multilayers
import numpy as np
from numba import jit


def stacking(g):
    """Return the array with the stacking for a gemetry"""
    rs = g.multireplicas(1) # replicated
    return circle_overlap(g.r,rs)


def circle_overlap(rs0,rs1):
    """Return the circle overlap for a set of locations"""
    out = []
    for r in rs0:
        z = r[2] # get the z location 
        dzs = np.abs(rs1[:,2] - z) # distance to the closest layer
        dz = np.min(dzs[dzs>1e-5]) # closest distance
        cls = np.abs(dzs - dz) # this will be zero for the closest layer
        rs = rs1[cls<1e-5] # select atoms of the closest layers
        rs = rs - r # shift by the specific atom
        rs = rs[:,0:2] # retain only xy coordinates
        drs = np.sum(rs*rs,axis=1) # compute distance
        drs = drs[drs<(10.)] # only closest sites
        o = circle_overlap_jit(drs) # compute the overlap
        out.append(o) # store
    return np.array(out) # return the stacking

@jit(nopython=True)
def circle_overlap_jit(drs):
    out = 0. # initialize
    for dr in drs: # loop over distances
        out += np.exp(-dr) # this is just a workaround
    return out
