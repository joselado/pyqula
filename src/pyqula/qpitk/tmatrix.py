import numpy as np
from scipy.sparse import issparse


def impurity_matrix(h,impurity):
    """Return the impurity Hamiltonian V of one unit cell as a dense
    matrix of the dimension of h.intra.

    impurity is either a number, an onsite potential on the first site of
    the cell, an array with one onsite potential per site of the cell, or
    a matrix of the dimension of h.intra (for a magnetic, a spin-orbit or
    a Nambu impurity). The onsite potentials go through h.spinless2full,
    so they are spin independent and carry the tau_z sign of a potential
    in a Nambu Hamiltonian"""
    n = h.intra.shape[0] # dimension of the cell
    ns = len(h.geometry.r) # number of sites
    if impurity is None:
        raise ValueError("get_qpi(mode='tmatrix') needs an impurity: a "
                "number (an onsite potential on the first site), one "
                "onsite potential per site of the cell, or a matrix of "
                "the dimension of the cell, "+str(n))
    if issparse(impurity): impurity = impurity.toarray()
    v = np.array(impurity,dtype=np.complex128)
    if v.ndim==0: # onsite potential on the first site
        v = np.zeros(ns,dtype=np.complex128) ; v[0] = impurity
    if v.ndim==1: # one onsite potential per site
        if len(v)!=ns:
            raise ValueError("the impurity of get_qpi(mode='tmatrix') "
                    "has "+str(len(v))+" onsite potentials, but the cell "
                    "has "+str(ns)+" sites")
        m = h.spinless2full(np.diag(v))
        if issparse(m): m = m.toarray()
        return np.array(m,dtype=np.complex128)
    if v.shape!=(n,n):
        raise ValueError("the impurity matrix of get_qpi(mode='tmatrix') "
                "has shape "+str(v.shape)+", but the cell has dimension "
                +str(n))
    return v


def electron_weights(h):
    """Weight of each component in the local density of states: one for
    every component of a normal Hamiltonian, and only the electron part
    of a Nambu one, which is what STM measures"""
    from ..operators import get_electron
    p = get_electron(h)
    if issparse(p): p = p.toarray()
    return np.real(np.diag(p))


def component_positions(h):
    """Position of the site of each component of the cell (spin and
    Nambu components share the position of their site)"""
    r = np.array(h.geometry.r)
    n = h.intra.shape[0]
    return r[np.arange(n)//(n//len(r))]


def reciprocal_vectors(g):
    """Cartesian reciprocal vectors b1,b2 with a_i.b_j=delta_ij, so a
    reduced momentum (q1,q2) is the Cartesian 2*pi*(q1*b1+q2*b2)"""
    a = np.array([g.a1[0:2],g.a2[0:2]])
    b = np.linalg.inv(a).T # rows are b1 and b2
    return b


def bloch_green_mesh(h,nk,energy,delta):
    """Bloch Green's function G0(k,w+i*delta) on the nk x nk mesh,
    (nk,nk,n,n), and the clean spectral function -Im Tr G0/pi on it"""
    hk = h.get_hk_gen()
    ks = np.arange(nk)/nk
    hks = np.array([[_dense(hk([k1,k2,0.])) for k2 in ks] for k1 in ks])
    (es,us) = np.linalg.eigh(hks) # diagonalize every k
    d = 1./(energy+1j*delta-es) # the poles
    g = np.einsum("xyab,xyb,xycb->xyac",us,d,np.conj(us))
    return g


def _dense(m):
    if issparse(m): return m.toarray()
    return np.array(m)


def tmatrix_qpi_lattice(h,v,energy,nk=20,delta=0.1,born=False):
    """Change of the local density of states of every component a of the
    cell at cell n of the nk x nk torus, drho_a(n), for one impurity v
    in the cell n=0, and its Fourier transform drho_a(q) on the mesh
    q=(m1,m2)/nk. Also returns the T-matrix and the clean spectral
    function on the k-mesh.

    With G0(n) = (1/N) sum_k exp(-2 pi i k.n) G0(k), the Fourier
    coefficient of the Bloch Green's function, the T-matrix of a
    perturbation confined to the cell n=0 does not depend on momentum,
        T(w) = V [1 - G0(0) V]^-1 ,
    and the Dyson equation G = G0 + G0 T G0 gives
        drho_a(n) = -(1/pi) Im [G0(-n) T G0(n)]_aa ,
    whose Fourier transform is Eq. (36)-(37) of arXiv:2607.22487 with
    the k-sum done on the same mesh. The sum is done in real space with
    two FFTs, which is exact on the torus"""
    w = electron_weights(h)
    g = bloch_green_mesh(h,nk,energy,delta)
    akw = -np.imag(np.einsum("xyaa,a->xy",g,w))/np.pi # clean spectral function
    gr = np.fft.fft2(g,axes=(0,1))/(nk*nk) # G0(n) for n=(i,j)
    g0 = gr[0,0] # local Green's function
    n = v.shape[0]
    if born: t = v.copy() # Born approximation, Eq. (31)
    else: t = v@np.linalg.inv(np.identity(n)-g0@v) # Eq. (24)
    s = np.where(np.sum(np.abs(t),axis=0)+np.sum(np.abs(t),axis=1)>0.)[0]
    ts = t[np.ix_(s,s)] # restricted to the support of the impurity
    grm = np.roll(np.flip(gr,axis=(0,1)),1,axis=(0,1)) # G0(-n)
    dg = np.einsum("xyab,bc,xyca->xya",grm[:,:,:,s],ts,gr[:,:,s,:])
    drho = -np.imag(dg)/np.pi*w[None,None,:] # change of the LDOS
    drhoq = np.fft.fft2(drho,axes=(0,1)) # sum_n exp(-2 pi i q.n) drho(n)
    return drho,drhoq,t,akw


def get_qpi_tmatrix(h,qs,energies,nk=20,delta=0.1,impurity=None,
        born=False):
    """QPI of a single impurity from the T-matrix, Eqs. (24), (36) and
    (37) of arXiv:2607.22487.

    qs are the momenta in reduced coordinates, which must be multiples of
    1/nk, energies the energies. Returns a list with one (|drho(q)|,A(q))
    pair per energy, where drho(q) = sum_a exp(-i q.tau_a) drho_a(q) is
    the Fourier transform of the change of the local density of states
    taken at the true position tau_a of each site, as in
    get_qpi_impurity, and A the clean spectral function at the same
    momentum folded into the first zone"""
    if h.dimensionality!=2:
        raise ValueError("get_qpi(mode='tmatrix') is only implemented for "
                "2d Hamiltonians")
    h = h.copy()
    h.turn_dense()
    v = impurity_matrix(h,impurity)
    qs = np.array(qs)
    m = np.rint(qs[:,0:2]*nk).astype(int) # index of each q on the mesh
    if np.max(np.abs(m/nk-qs[:,0:2]))>1e-6:
        raise ValueError("get_qpi(mode='tmatrix') needs the momenta on "
                "the k-mesh, multiples of 1/nk with nk="+str(nk))
    b = reciprocal_vectors(h.geometry)
    qc = 2.*np.pi*qs[:,0:2]@b # Cartesian momenta
    tau = component_positions(h)[:,0:2]
    phase = np.exp(-1j*qc@tau.T) # (nq,n), exp(-i q.tau_a)
    i1,i2 = m[:,0]%nk,m[:,1]%nk
    out = []
    for e in energies:
        drho,drhoq,t,akw = tmatrix_qpi_lattice(h,v,e,nk=nk,delta=delta,
                born=born)
        dq = np.sum(phase*drhoq[i1,i2,:],axis=1)
        out.append((np.abs(dq),akw[i1,i2]))
    return out


def tmatrix_qmesh(nk,nsuper):
    """Momenta in reduced coordinates on the k-mesh, multiples of 1/nk,
    covering nsuper x nsuper Brillouin zones centered at Gamma"""
    ms = np.arange(nk*nsuper) - (nk*nsuper)//2
    return np.array([[m1/nk,m2/nk,0.] for m1 in ms for m2 in ms])
