"""Nested Wilson loop: the polarization of a sector of Wannier bands

The Wilson loop along the first reciprocal direction has the hybrid Wannier
centers nu_x(k_y) as the phases of its eigenvalues, the Wannier bands. When
they split into sectors separated by a gap, the eigenvectors of the loop in
one sector span a subspace of the occupied states at each k, and a second
Wilson loop of that subspace along the other direction, the nested Wilson
loop, gives the polarization along y of the electrons in that sector
(Benalcazar, Bernevig and Hughes, arXiv:1611.07987, with the algorithm of
section VI of arXiv:1708.04230, whose equation numbers are quoted below).
With the mirrors M_x and M_y each sector polarization is 0 or 1/2, and
their products give the quadrupole moment q_xy (eq. VI.47).

The states are written as the columns of a matrix, and the link between two
consecutive momenta is F_k = <u_{k+dk}|u_k>, replaced by the unitary part of
its polar decomposition, so that the loop F_{k+(N-1)dk}...F_{k+dk}F_k has
eigenvalues exp(i 2 pi nu) with nu the position of the center (eq. VI.12).
"""
import numpy as np
from scipy.linalg import schur
from .. import algebra


def _unitary_part(m):
    """Unitary part of the polar decomposition of a square matrix"""
    (u,s,vh) = np.linalg.svd(m)
    return u@vh


def _occupied_grid(h,nk,nocc,gauge):
    """Occupied states on an nk x nk grid of the Brillouin zone, and the
    periodic images

    Returns s, with the states at the fractional momentum (i/nk,j/nk) as the
    columns of s[i,j], and the diagonals d[0], d[1] of the matrices that
    take the states at k onto the ones at k+b_1 and k+b_2. In the lattice
    gauge every orbital sits at the origin of its cell, the Bloch
    Hamiltonian is periodic and the images are the states themselves; in
    the atomic gauge the component of an orbital at the fractional position
    x carries exp(-i 2 pi k.x), and the image at k+b_a picks up
    exp(-i 2 pi x_a)."""
    from .qgt import _check_gauge,_orbital_fractions
    _check_gauge(gauge)
    if h.dimensionality!=2:
        raise ValueError("the nested Wilson loop needs a two-dimensional "
          +"Hamiltonian, and this one has dimensionality "
          +str(h.dimensionality))
    hkgen = h.get_hk_gen() # Bloch Hamiltonian generator
    ks = np.arange(nk)/nk # the loops close on the periodic image of k=0
    s = [[None for k1 in ks] for k0 in ks]
    for (i,k0) in enumerate(ks):
        for (j,k1) in enumerate(ks):
            (es,vs) = algebra.eigh(hkgen([k0,k1,0.]))
            n = np.sum(es<0.) if nocc is None else nocc
            s[i][j] = vs[:,:n] # the lowest n states, as columns
    sizes = set([sij.shape[1] for si in s for sij in si])
    if len(sizes)!=1:
        raise ValueError("the number of occupied states changes over the "
          +"Brillouin zone (between "+str(min(sizes))+" and "+str(max(sizes))
          +"), so the occupied manifold is not separated by a gap and its "
          +"Wannier bands are not defined. Shift the Fermi energy into a "
          +"gap with h.shift_fermi / h.set_filling first")
    s = np.array(s,dtype=np.complex128) # (nk,nk,orbitals,occupied)
    norb = s.shape[2]
    if gauge=="atomic":
        frac = _orbital_fractions(h,norb) # (orbitals,2)
        for (i,k0) in enumerate(ks):
            for (j,k1) in enumerate(ks):
                phase = np.exp(-2j*np.pi*frac@np.array([k0,k1]))
                s[i,j] = phase[:,None]*s[i,j]
        d = [np.exp(-2j*np.pi*frac[:,a]) for a in range(2)]
    else: d = [np.ones(norb),np.ones(norb)]
    return (s,d)


def _wannier_bands(s,d):
    """Wannier centers and Wilson-loop eigenvectors at every base point

    s[i,j] holds the states with the loop running along i, and d is the
    diagonal of the periodic image along it. Returns nu[i,j], the centers
    in (-1/2,1/2], and v[i,j], whose columns are the eigenvectors of the
    Wilson loop with base point (i,j), in the basis of the columns of
    s[i,j]. The loop at base i+1 is F_i W_i F_i^dagger, so each line needs
    one product of its links rather than one per base point."""
    (nk,nl) = s.shape[0:2]
    nocc = s.shape[3]
    nu = np.zeros((nk,nl,nocc))
    v = np.zeros((nk,nl,nocc,nocc),dtype=np.complex128)
    for j in range(nl): # every line of the loop direction
        line = [s[i,j] for i in range(nk)] + [d[:,None]*s[0,j]]
        links = [_unitary_part(line[i+1].conj().T@line[i]) for i in range(nk)]
        w = np.identity(nocc,dtype=np.complex128)
        for f in links: w = f@w # loop with base point 0
        for i in range(nk):
            (t,q) = schur(w,output="complex") # unitary, so t is diagonal
            nu[i,j] = np.angle(np.diag(t))/(2.*np.pi)
            v[i,j] = q
            w = links[i]@w@links[i].conj().T # move the base point to i+1
    return (nu,v)


def _sector_polarizations(s,d,loop,select,tol=None):
    """Polarization along the other direction of a sector of the Wannier
    bands of the loop along direction loop, at each momentum along loop

    select takes an array of Wannier centers and returns a boolean mask of
    the ones in the sector. Returns the polarizations p(k) (eq. VI.14) and
    the Wannier centers nu[i,j] at every base point, with i along loop.
    With a tol, the Wannier gap around the lines 0 and 1/2 is checked
    first, see _check_wannier_gap."""
    other = 1 - loop
    if loop==1: s = s.transpose(1,0,2,3) # the loop direction goes first
    (nu,v) = _wannier_bands(s,d[loop])
    if tol is not None: _check_wannier_gap(nu,tol)
    mask = select(nu)
    counts = np.sum(mask,axis=2)
    if counts.min()!=counts.max() or counts.max()==0:
        raise ValueError("the Wannier sector holds between "+str(counts.min())
          +" and "+str(counts.max())+" Wannier bands over the Brillouin "
          +"zone, so it is not separated from the rest by a Wannier gap "
          +"and its polarization is not defined")
    nk = s.shape[0]
    p = np.zeros(nk)
    for i in range(nk): # the nested loop, along the other direction
        w = [s[i,j]@v[i,j][:,mask[i,j]] for j in range(nk)] # eq. VI.5
        w.append(d[other][:,None]*w[0]) # the periodic image closes it
        wt = np.identity(w[0].shape[1],dtype=np.complex128)
        for j in range(nk): wt = _unitary_part(w[j+1].conj().T@w[j])@wt
        p[i] = np.angle(np.linalg.det(wt))/(2.*np.pi) # eq. VI.14
    return (p,nu)


def _average(p):
    """Average of a polarization defined modulo 1 over a closed loop of
    momenta (eq. VI.15), which must come back to itself"""
    q = np.unwrap(2.*np.pi*p)/(2.*np.pi)
    close = (p[0] - q[-1] + 0.5)%1. - 0.5 # the step that closes the loop
    if abs(q[-1] + close - q[0])>0.5: # it wound around
        raise ValueError("the polarization of the Wannier sector winds "
          +"by "+str(int(np.round(q[-1]+close-q[0])))+" across the "
          +"Brillouin zone, so the sector has a Chern number and its "
          +"polarization is not defined")
    return np.mean(q)


def _in_range(p):
    """p modulo 1 in [-1/4,3/4), so that neither of the quantized values 0
    and 1/2 sits at an edge of the interval, where roundoff would move it
    to the other end"""
    return (p + 0.25)%1. - 0.25


# sector= of the polarization -> the Wannier centers it keeps, in (-1/2,1/2]
_sectors = {
    "+": lambda nu: (nu>0.) & (nu<0.5),
    "-": lambda nu: nu<0.,
    }


def _wannier_gap(nu):
    """Smallest distance of any Wannier center from the two lines 0 and 1/2
    that separate the sectors"""
    return np.min(np.minimum(np.abs(nu),0.5-np.abs(nu)))


def _check_sector(sector):
    if sector not in _sectors:
        raise ValueError("unknown sector '"+str(sector)+"'; it must be one "
          +"of "+str(list(_sectors)))


def _check_wannier_gap(nu,tol):
    """Raise if a Wannier center comes within tol of the lines 0 and 1/2
    that separate the sectors "+" and "-" """
    gap = _wannier_gap(nu)
    if gap<tol:
        raise ValueError("the Wannier bands come within "+str(gap)+" of the "
          +"line 0 or 1/2 that separates the two sectors, below the "
          +"tolerance "+str(tol)+", so the sectors are not resolved: the "
          +"Wannier gap closes, or nearly closes, which can happen without "
          +"any closing of the bulk gap. Lower tol to accept it anyway")


def _sector_polarization(s,d,loop,sector,tol):
    """Polarization of a sector, refused if the Wannier gap is below tol"""
    (p,nu) = _sector_polarizations(s,d,loop,_sectors[sector],tol=tol)
    return _in_range(_average(p))


def wannier_sector_polarization(h,loop=0,sector="+",nk=40,nocc=None,
        gauge="lattice",tol=1e-4):
    """Polarization of a sector of the Wannier bands, from the nested
    Wilson loop

    The Wilson loop along the reciprocal direction loop gives the Wannier
    bands nu(k) at each momentum k along the other direction; the sector
    "+" keeps the ones with 0 < nu < 1/2 and "-" the ones with
    -1/2 < nu < 0, which need a Wannier gap at 0 and 1/2, as the mirror of
    the loop direction guarantees. The nested Wilson loop of the sector
    along the other direction gives its polarization p(k') at each momentum
    k' along loop, and the result is their average, the p_y^{nu_x^+} of
    Benalcazar, Bernevig and Hughes for loop=0 and sector="+" (eq. VI.15 of
    arXiv:1708.04230). It is a position along the other lattice vector in
    units of it, returned modulo 1 in [-1/4,3/4), and it is 0 or 1/2 with a
    mirror symmetry of the other direction.

    nk: momenta along each direction
    nocc: the number of bands, counted from the lowest, that make the
        occupied manifold; by default the ones below zero energy
    gauge: "lattice", every orbital at the origin of its cell, which is the
        convention of Benalcazar, Bernevig and Hughes, or "atomic", every
        orbital at its position, which moves the result with the positions
    tol: the smallest distance of the Wannier bands from the lines 0 and
        1/2 that is accepted; below it this raises, since the model can
        change phase through a closing of the Wannier gap alone"""
    _check_sector(sector)
    (s,d) = _occupied_grid(h,nk,nocc,gauge)
    return _sector_polarization(s,d,loop,sector,tol)


def wannier_gap(h,loop=0,nk=40,nocc=None):
    """Smallest distance of the Wannier bands of the loop along loop from
    the lines 0 and 1/2 that separate the two sectors of
    wannier_sector_polarization; the sectors are defined while it stays
    finite"""
    (s,d) = _occupied_grid(h,nk,nocc,"lattice")
    if loop==1: s = s.transpose(1,0,2,3)
    (nu,v) = _wannier_bands(s,d[loop])
    return _wannier_gap(nu)


def quadrupole_moment(h,nk=40,nocc=None,tol=1e-4):
    """Quadrupole moment q_xy of a two-dimensional insulator with the mirrors
    M_x and M_y, from the Wannier-sector polarizations

    q_xy = p_y^{nu_x^-} p_x^{nu_y^-} + p_y^{nu_x^+} p_x^{nu_y^+} modulo 1
    (eq. VI.47 of arXiv:1708.04230), which is 1/2 when both nested
    polarizations are 1/2 and 0 otherwise. It is quantized only when the
    two mirrors are symmetries, and is returned in [-1/4,3/4) so that
    neither 0 nor 1/2 sits at an edge; the result is in the lattice gauge,
    as in that paper. Raises if either Wannier gap is below tol."""
    (s,d) = _occupied_grid(h,nk,nocc,"lattice")
    p = {}
    for loop in (0,1):
        for sector in ("+","-"):
            p[(loop,sector)] = _sector_polarization(s,d,loop,sector,tol)
    q = p[(0,"-")]*p[(1,"-")] + p[(0,"+")]*p[(1,"+")]
    return _in_range(q)
