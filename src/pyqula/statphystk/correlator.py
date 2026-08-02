import numpy as np
from numba import jit

def get_nnc(g,den,n=20,normalized=False,**kwargs): 
    """Compute the first N correlators"""
    g0 = g.copy() ; g0.dimensionality = 0 # zero dimensional
    ds = g0.neighbor_distances() # neighbor distances
    if len(ds)<n: n = len(ds)
    r = g0.r # locations
    ri = r.copy() # copy locations
    deni = den.copy()
    if g.dimensionality>0: # periodic boundaries
        rj = g.multireplicas(1) # replicas of the locations
        den0 = den.copy() # copy
        denj = den0.copy() # start
        for i in range(1,len(rj)//len(den0)): denj = np.concatenate([denj,den0])
    else:
        rj = ri
        denj = deni
    cs = [get_nnci_jit(ri,rj,deni,denj,di,1e-3) for di in ds[0:n]]
    cs = np.array(cs) # convert to array
    if normalized: # normalize using Cauchy-Swartz inequality
        var = np.mean((den - np.mean(den))**2)
        # ignore replica effect due to periodic BC
        cs = cs/var # normalize by variance
    x,y = ds[0:len(cs)],cs # distance and correlators
    return x,y











@jit(nopython=True)
def get_nnci_jit_v1(ri,rj,deni,denj,di,delta):
    """Compute a single correlator"""
    out = 0. # output value
    no = 0. # counter
    outi,outj,noi,noj = 0.,0.,0.,0.
    for i1 in range(len(ri)):
        for i2 in range(len(rj)):
            r1 = ri[i1]
            r2 = rj[i2]
            dr = r1-r2
            dr2 = np.sum(dr*dr)
            if np.abs(dr2 - di**2)<delta:
                out += deni[i1]*denj[i2]
                no += 1.
                outi += deni[i1]
                outj += denj[i2]
                noi += 1
                noj += 1
    if no==0: return 0.
    return out/no  - outi/noi*outj/noj # return correlator








@jit(nopython=True)
def get_nnci_jit(ri,rj,deni,denj,di,delta):
    """Compute a single correlator"""
    out = 0. # output value
    no = 0. # counter
    deni = deni - np.mean(deni) # redefine so that it has zero average
    denj = denj - np.mean(denj) # redefine so that it has zero average
    for i1 in range(len(ri)):
        for i2 in range(len(rj)):
            r1 = ri[i1]
            r2 = rj[i2]
            dr = r1-r2
            dr2 = np.sum(dr*dr)
            if np.abs(dr2 - di**2)<delta:
                out += deni[i1]*denj[i2]
                no += 1.
    if no==0: return 0. # no neighbor found
    return out/no  #- np.mean(deni)*np.mean(denj) # return correlator



def get_structure_factor(g,den,qpath=None,nq=60,qmax=None):
    """Static structure factor S(q)=|sum_i (den_i-mean(den)) e^{-iq.r_i}|^2/N
    of the current occupation snapshot, evaluated directly on the
    real-space positions g.r (no periodic replicas, unlike get_nnc):
    the reciprocal-space companion to get_nnc() -- where the
    neighbor-shell correlator G(r) tells you the ordering length
    scale, S(q) tells you the ordering wavevector. Subtracting the
    mean makes S(q=0)=0 identically, so a peak elsewhere in q is what
    signals order

    qpath: explicit array of q vectors to evaluate; if None, a default
    square grid of `nq`x`nq` points spanning +-qmax is used, with qmax
    set from the nearest-neighbor spacing (2*pi/d) unless given"""
    r = g.r
    n = len(den)
    if qpath is None:
        if qmax is None:
            ds = g.neighbor_distances() # shell distances
            qmax = 2.*np.pi/ds[0] # set by the nearest-neighbor spacing
        qs = np.linspace(-qmax,qmax,nq)
        qpath = np.array([[qx,qy,0.] for qx in qs for qy in qs])
    else:
        qpath = np.array(qpath)
    den0 = den - np.mean(den) # q=0 removed, so S(q=0)=0 identically
    sq = _structure_factor_jit(r[:,0],r[:,1],r[:,2],den0,
            qpath[:,0],qpath[:,1],qpath[:,2])
    return qpath,sq


@jit(nopython=True)
def _structure_factor_jit(rx,ry,rz,den0,qx,qy,qz):
    """Core double loop for get_structure_factor: |sum_i den0_i e^{-iq.r_i}|^2/N"""
    n = len(den0)
    nq = len(qx)
    out = np.zeros(nq)
    for iq in range(nq):
        sr = 0. ; si = 0. # real/imaginary parts of sum_i den0_i e^{-iq.r_i}
        for i in range(n):
            phase = rx[i]*qx[iq]+ry[i]*qy[iq]+rz[i]*qz[iq]
            sr += den0[i]*np.cos(phase)
            si -= den0[i]*np.sin(phase)
        out[iq] = (sr*sr+si*si)/n
    return out
