# Kubo-Greenwood optical conductivity of a periodic tight-binding
# Hamiltonian. The physics, the units and the sign conventions are
# documented in the public entry point, conductivity.py -- read that
# first; this module only holds the machinery.
import numpy as np
from numba import jit, prange

from .. import algebra
from .. import current
from .. import klist
from .. import parallel


def _setup(h):
    """Return everything that only depends on the Hamiltonian (not on k):
    a multicell copy hm of h, the derivative "orders" (one per periodic
    direction, in multicell.derivative's convention), the Bloch generator
    hkgen, the reduced->Cartesian Jacobian jac (see _velocities), the
    intracell bond vectors dr (see _bond_vectors), the unit cell volume
    and a characteristic hopping energy scale (used to make
    degeneracy_tol relative rather than absolute, as in topologytk/qgt.py).

    As in topologytk/qgt.py's _multicell_and_orders: a .copy() is taken
    because get_multicell() may hand back h itself, and every stored
    matrix is coerced to a plain numpy.ndarray because Hamiltonians built
    through get_supercell() keep their hoppings as the legacy
    numpy.matrix, whose "*" is a matrix product -- that would silently
    corrupt the elementwise algebra downstream. hkgen is built once here
    and reused at every k-point (rebuilding it per k-point does real setup
    work and is dramatically slower on a mesh sweep)."""
    hm = h.get_multicell().copy() # own copy: get_multicell() may alias h
    hm.intra = np.asarray(hm.intra)
    for t in hm.hopping: t.m = np.asarray(t.m)
    dim = h.dimensionality
    g = h.geometry
    if dim==1:
        orders = [[1]]
        avecs = np.array([g.a1])
        cellvol = np.sqrt(g.a1.dot(g.a1)) # length of the unit cell
    elif dim==2:
        orders = [[1,0],[0,1]]
        avecs = np.array([g.a1,g.a2])
        cellvol = np.linalg.norm(np.cross(g.a1,g.a2)) # area of the unit cell
    else: raise NotImplementedError(
        "optical conductivity only implemented for dimensionality 1 or 2 "
        "(current.derivative, the shared k-derivative, has no 3D branch)")
    hkgen = hm.get_hk_gen() # build once, reuse at every k-point
    scale = max(np.max(np.abs(hm.intra)),
                max((np.max(np.abs(t.m)) for t in hm.hopping),default=0.0),
                1e-12) # characteristic hopping energy (floored, so that a
                       # Hamiltonian with vanishing hoppings still gets a
                       # small but nonzero scale)
    jac = np.array(avecs)/(2.*np.pi) # jac[i,alpha] = a_i[alpha]/(2 pi)
    dr = _bond_vectors(h,hm.intra.shape[0])
    return hm,orders,hkgen,jac,dr,cellvol,scale


def _bond_vectors(h,n):
    """Intracell bond vectors dr[alpha,i,j] = r_j[alpha] - r_i[alpha]
    between the orbitals of the unit cell, with r taken from the geometry
    (the standard tight-binding approximation of a position operator
    diagonal in the orbital basis). See _velocities for what they are for.

    pyqula orders the basis as (site,spin) -- and (site,spin,electron-hole)
    for Nambu -- with the site index slowest, so the site positions simply
    repeat n/len(geometry.r) times. Nambu Hamiltonians are rejected: there
    the charge vertex is not the identity (the hole block carries the
    opposite charge), so the current operator is not i[H,r] and this
    module's formula does not apply as written."""
    if h.has_eh: raise NotImplementedError(
        "optical conductivity is not implemented for Nambu (superconducting) "
        "Hamiltonians: the charge current operator there is not i[H,r]")
    r = np.array(h.geometry.r)
    nsites = len(r)
    if n%nsites!=0: raise ValueError(
        "Hamiltonian dimension is not a multiple of the number of sites")
    rr = np.repeat(r,n//nsites,axis=0) # one position per orbital
    return np.array([rr[None,:,a]-rr[:,None,a] for a in range(3)])


def _velocities(hm,orders,jac,dr,hk,k):
    """Cartesian velocity operators v_alpha = i[H,r_alpha] (alpha = x,y,z)
    of the Bloch Hamiltonian at the reduced-coordinate k-point k, with
    hbar = 1 (so v carries units of energy times length).

    Two ingredients, both necessary:

    1) current.hk_derivative gives dH/dk_i with respect to the *reduced*
    (crystal) coordinate k_i -- it is the shared, correctly-normalized
    wrapper around multicell.derivative, whose raw output is short by a
    factor 2*pi per derivative order (see its docstring; derivative()
    itself must not be touched, other callers compensate locally). Since
    the Bloch phase is exp(2 pi i k.R) with R = sum_i R_i a_i, the crystal
    momentum is K = sum_i k_i b_i with a_i.b_j = 2 pi delta_ij, so
    k_i = a_i.K/(2 pi) and the chain rule gives

        dH/dK_alpha = sum_i (a_i[alpha]/(2 pi)) dH/dk_i = sum_i jac[i,alpha] dH/dk_i

    which is what jac (from _setup) encodes.

    2) pyqula builds H(k) in the "lattice" (Wannier90 convention I) gauge,
    where the Bloch phase carries only the lattice vector R and not the
    intracell offset r_j - r_i. dH/dK is therefore *not* the velocity: it
    misses every intracell bond, so e.g. on the honeycomb lattice (whose
    two sites sit at +-x/2) it would leave the x direction with only two
    of the three nearest-neighbour bonds and break the C3 symmetry of the
    answer, giving sigma_xx != sigma_yy. The Peierls substitution
    t_ij -> t_ij exp(i A.d_ij) with the *full* bond vector d_ij = R + r_j
    - r_i gives the correct current operator, and the missing piece is
    exactly a commutator with the (diagonal) position operator:

        v_alpha = dH/dK_alpha + i [H(k), r_alpha] ,

    elementwise i H_ij (r_j - r_i)_alpha, i.e. 1j*hk*dr[alpha] here. This
    is Wannier90/PythTB's "atomic" gauge (convention II) for the velocity
    matrix elements, evaluated in pyqula's lattice-gauge eigenbasis, which
    is legitimate because the two gauges differ by the diagonal unitary
    exp(-i K.r_j) that relates the two Hamiltonians. The correction does
    not touch the diagonal (band velocity) elements -- <n|[H,r]|n> = 0 --
    so it leaves the Drude weight alone and only changes the interband
    (optical) matrix elements.

    The normalization of the whole chain is pinned by analytic benchmarks
    in tests/conductivity: the 1D chain (dE/dK = -2 t sin K, f-sum-rule
    weight 2|t|/pi at half filling) and the universal optical conductivity
    of graphene, pi e^2/(4 h) per spin."""
    dhs = [current.hk_derivative(hm,k,order=o) for o in orders]
    n = hk.shape[0]
    v = np.zeros((3,n,n),dtype=np.complex128)
    for alpha in range(3): # loop over Cartesian directions
        for i in range(len(orders)): # loop over periodic directions
            v[alpha] = v[alpha] + jac[i,alpha]*dhs[i]
        v[alpha] = v[alpha] + 1j*hk*dr[alpha] # intracell bonds
    return v


def _second_derivatives(hm,orders,jac,dr,hk,k):
    """Cartesian second derivatives d2H/dK_a dK_b of the Peierls-coupled
    Hamiltonian, i.e. the inverse effective mass operator that enters the
    f-sum rule (see sum_rule_weight). Applying the same "full bond vector"
    rule twice (see _velocities: the velocity multiplies every hopping by
    i*d_ij, so the second derivative multiplies it by -d_ij d_ij) gives

      d2H/dK_a dK_b = d2H/dK_a dK_b|lattice + i (dH/dK_a) o dr_b
                      + i (dH/dK_b) o dr_a - H o dr_a o dr_b

    with "o" an elementwise (Hadamard) product. Returns a 3x3 nested list
    of matrices."""
    dim = len(orders)
    # reduced-coordinate first and second derivatives
    d1 = [current.hk_derivative(hm,k,order=o) for o in orders]
    d2 = [[None for j in range(dim)] for i in range(dim)]
    for i in range(dim):
        for j in range(i,dim):
            order = [orders[i][d]+orders[j][d] for d in range(dim)]
            d2[i][j] = current.hk_derivative(hm,k,order=order)
            d2[j][i] = d2[i][j] # symmetric in (i,j)
    # Cartesian lattice-gauge derivatives, one per Cartesian direction
    dK = [sum(jac[i,a]*d1[i] for i in range(dim)) for a in range(3)]
    out = [[None for b in range(3)] for a in range(3)]
    for a in range(3):
        for b in range(3):
            m = sum(jac[i,a]*jac[j,b]*d2[i][j]
                    for i in range(dim) for j in range(dim))
            out[a][b] = m + 1j*dK[a]*dr[b] + 1j*dK[b]*dr[a] - hk*dr[a]*dr[b]
    return out


def _kmesh(h,nk):
    """Uniform Gamma-centered k-mesh of the Brillouin zone. nk is the
    number of points *per periodic direction*, so a 2D mesh holds nk*nk
    points."""
    return klist.kmesh(h.dimensionality,nk=nk)


def _hk(hkgen,k):
    """Bloch Hamiltonian at k, as a plain dense array (the elementwise
    products in _velocities require an ndarray, not a sparse matrix or the
    legacy numpy.matrix)"""
    return np.asarray(algebra.todense(hkgen(k)))


def _bands_and_velocities(h,ks):
    """Diagonalize H(k) on every k-point of ks and rotate the Cartesian
    velocity operators into the instantaneous eigenbasis. Returns the band
    energies es[k,n], the velocity matrix elements vs[k,alpha,n,m] =
    <n|v_alpha|m>, the unit cell volume and the Hamiltonian's energy
    scale."""
    hm,orders,hkgen,jac,dr,cellvol,scale = _setup(h)
    nk = len(ks)
    n = hm.intra.shape[0]
    es = np.zeros((nk,n),dtype=np.float64)
    vs = np.zeros((nk,3,n,n),dtype=np.complex128)
    for ik in range(nk):
        k = ks[ik]
        hk = _hk(hkgen,k)
        (e,w) = algebra.eigh(hk) # w[:,n] is the eigenvector of e[n]
        wc = np.conjugate(w)
        v = _velocities(hm,orders,jac,dr,hk,k)
        es[ik] = e
        for alpha in range(3):
            vs[ik,alpha] = wc.T@v[alpha]@w # <n|v_alpha|m>
    return es,vs,cellvol,scale


def _fermi(es,T):
    """Fermi occupations at temperature T. The chemical potential is
    always zero in pyqula (use h.shift_fermi to move the Fermi level).
    T<=0 gives the sharp zero-temperature step."""
    if T<=0.: return np.where(es<0.,1.,0.)
    return 1./(1.+np.exp(np.clip(es/T,-500.,500.)))


def _minus_dfermi(es,T):
    """-df/dE, the (normalized) thermal weight of the Fermi surface. At
    T<=0 this is a Dirac delta, which cannot be represented on a discrete
    k-mesh, so it is returned as zero -- i.e. the intraband/Drude channel
    silently vanishes at exactly T=0 and a positive temperature is
    required for it (see conductivity.optical_conductivity)."""
    if T<=0.: return np.zeros(es.shape)
    f = _fermi(es,T)
    return f*(1.-f)/T


def _response_weights(es,T,tol,intraband=True,interband=True):
    """Occupation weights of the Kubo formula, ratio[k,n,m], together with
    the transition energies dE[k,n,m] = E_m - E_n.

    ratio is (f_n - f_m)/(E_m - E_n), the factor that multiplies
    v^a_nm v^b_mn / (omega + i delta - (E_m - E_n)) in the Kubo formula.
    Its m -> n limit is -df/dE(E_n), which is exactly the intraband/Drude
    channel: computing the degenerate (|E_m - E_n| < tol) entries with that
    limit rather than with the raw quotient is what lets a single sum over
    *all* band pairs cover the interband and the Drude term at once, and
    is also what keeps a (near-)degenerate multiplet -- a spin-degenerate
    pair, a band crossing -- from blowing up on 0/0. intraband/interband
    switch the two families of terms on and off independently."""
    f = _fermi(es,T)
    dE = es[:,None,:] - es[:,:,None] # dE[k,n,m] = E_m - E_n
    df = f[:,:,None] - f[:,None,:] # df[k,n,m] = f_n - f_m
    small = np.abs(dE)<tol # (near-)degenerate pairs, including n==m
    safe = np.where(small,1.,dE) # avoid a 0/0 warning on the small entries
    fp = np.broadcast_to(_minus_dfermi(es,T)[:,:,None],dE.shape)
    ratio = np.where(small,fp,df/safe)
    if not intraband: ratio = np.where(small,0.,ratio)
    if not interband: ratio = np.where(small,ratio,0.)
    return ratio,dE


@jit(nopython=True,parallel=True,cache=True)
def _sigma_jit(dE,ratio,vs,omegas,delta):
    """Sum of the Kubo-Greenwood formula over k-points and band pairs, for
    every frequency and every Cartesian pair (a,b). The k-sum is *not*
    normalized here (the caller divides by nk times the cell volume) and
    the overall factor i is applied at the end.

    The parallel loop runs over frequencies rather than over k-points so
    that each thread owns one output slice out[iw] -- no reduction and no
    race, and no (nk,nomega,3,3) temporary, which would dominate the
    memory footprint on a fine mesh."""
    nk = dE.shape[0]
    n = dE.shape[1]
    nw = omegas.shape[0]
    out = np.zeros((nw,3,3),dtype=np.complex128)
    for iw in prange(nw): # parallel loop over frequencies
        acc = np.zeros((3,3),dtype=np.complex128)
        w = omegas[iw]
        for ik in range(nk): # loop over k-points
            for i in range(n): # loop over bands
                for j in range(n): # loop over bands
                    r = ratio[ik,i,j]
                    if r==0.: continue # switched off or unoccupied pair
                    pref = r/(w + 1j*delta - dE[ik,i,j])
                    for a in range(3): # Cartesian component
                        va = vs[ik,a,i,j] # <i|v_a|j>
                        if va==0.: continue
                        for b in range(3): # Cartesian component
                            acc[a,b] = acc[a,b] + pref*va*vs[ik,b,j,i]
        out[iw] = acc
    return 1j*out


def optical_conductivity(h,energies=None,nk=20,T=None,delta=0.1,
        intraband=True,interband=True,component=None,degeneracy_tol=1e-6):
    """Kubo-Greenwood optical conductivity tensor, see
    conductivity.optical_conductivity for the documentation."""
    if energies is None: energies = np.linspace(0.,4.,100)
    energies = np.array(energies,dtype=np.float64)
    if T is None: T = delta # same default as chitk/chiAB.py
    ks = _kmesh(h,nk)
    es,vs,cellvol,scale = _bands_and_velocities(h,ks)
    ratio,dE = _response_weights(es,T,degeneracy_tol*scale,
            intraband=intraband,interband=interband)
    parallel.set_num_threads() # set the number of threads for numba
    sigma = _sigma_jit(dE,ratio,vs,energies,delta)
    sigma = sigma/(len(ks)*cellvol) # (1/V) sum_k, with V = N_k * V_cell
    if component is not None:
        return energies,sigma[:,_index(component[0]),_index(component[1])]
    return energies,sigma


def _index(c):
    """Cartesian index of a component label ("x", "y" or "z")"""
    d = {"x":0,"y":1,"z":2}
    if c not in d: raise ValueError("Unknown Cartesian component "+str(c))
    return d[c]


def drude_weight(h,nk=20,T=None,degeneracy_tol=1e-6):
    """Drude (intraband) weight tensor, see conductivity.drude_weight"""
    if T is None: T = 0.05
    ks = _kmesh(h,nk)
    es,vs,cellvol,scale = _bands_and_velocities(h,ks)
    ratio,dE = _response_weights(es,T,degeneracy_tol*scale,
            intraband=True,interband=False)
    # D_ab = (1/V) sum_k sum_{nm in a degenerate block} (-df/dE) v^a_nm v^b_mn
    # The trace over each degenerate block makes this real and basis
    # independent within the block (it is Tr[P v_a P v_b P], with P the
    # projector onto the block); the imaginary residue is numerical noise.
    D = np.einsum("knm,kanm,kbmn->ab",ratio,vs,vs,optimize=True)
    return D.real/(len(ks)*cellvol)


def sum_rule_weight(h,nk=20,T=None):
    """Diamagnetic (inverse effective mass) weight tensor, see
    conductivity.sum_rule_weight"""
    if T is None: T = 0.05
    hm,orders,hkgen,jac,dr,cellvol,scale = _setup(h)
    ks = _kmesh(h,nk)
    W = np.zeros((3,3),dtype=np.float64)
    for k in ks:
        hk = _hk(hkgen,k)
        (e,w) = algebra.eigh(hk)
        f = _fermi(e,T)
        wc = np.conjugate(w)
        d2 = _second_derivatives(hm,orders,jac,dr,hk,k)
        for a in range(3):
            for b in range(a,3):
                # diagonal expectation values <n|d2H/dK_a dK_b|n>
                di = np.real(np.einsum("in,ij,jn->n",wc,d2[a][b],w,
                        optimize=True))
                W[a,b] += np.dot(f,di)
    for a in range(3): # mirror the upper triangle, W is symmetric
        for b in range(a+1,3): W[b,a] = W[a,b]
    return W/(len(ks)*cellvol)
