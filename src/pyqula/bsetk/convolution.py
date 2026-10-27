"""The resonant BSE block at Q=0 applied as a convolution on the k-mesh.

factorize.KernelFactorization writes the direct term as one rank-one
projector per non-zero entry (a,b,d) of the real-space interaction, so its
matvec costs O(nterm npair) with nterm = norb^2 times the number of lattice
vectors inside the cutoff, and its factors take O(nterm npair) memory. The
same term is a convolution over the mesh. With the pairs n = (k,v,c),

    (D x)[k,v,c] = (1/N) sum_ab conj(C_c(k)_a) Y_ab(k) C_v(k)_b ,
    Y_ab(k)      = sum_k' W_ab(k - k') Z_ab(k') ,
    Z_ab(k)      = sum_vc C_c(k)_a x[k,v,c] conj(C_v(k)_b) ,

where C_c(k) and C_v(k) are the conduction and valence eigenvectors of the
pair (pb.el and pb.ho). Z is the pair vector written as a matrix in orbital
space, gathered once per k-point, and the convolution is done by FFT on the
mesh, with W tabulated at the mesh differences k - k'. That table is the
real-space dictionary folded onto the nk-periodic supercell, which is what
the Bloch phases of the factorization do as well, so the two applies are
equal to machine precision and not an approximation of each other. The
cost is O(npair norb + norb^2 N log N) per column and the memory
O(norb^2 N), independent of the range of the interaction, and a tabulated
(RPA-screened) W enters as it is, which the factorization has to refuse.

The exchange term at Q=0 needs no lattice sum at all: it is the orbital
diagonal of the same Z, summed over the mesh and contracted with W(Q=0),
and it is scattered back with the same projection.
"""
import numpy as np
import numba
from numba import njit, prange
import scipy.fft

from .interaction import interaction_at_q,interaction_at_qs
from .kernel import kernel_terms
from .. import parallel


class ConvolutionKernel():
    """The resonant block A = diag(dE) + X - D of a Q=0 pair basis, with
    the attributes of factorize.KernelFactorization that a matrix-free
    solver needs: npair, diag, nterm (zero only for the diagonal alone)
    and matvec"""
    def __init__(self,pb,W,Wx=None,kernel="full"):
        if Wx is None: Wx = W # exchange defaults to the direct interaction
        direct,exchange = kernel_terms(kernel)
        if np.any(np.abs(pb.Q)>1e-12): raise NotImplementedError(
                "the convolution apply is only implemented at Q=0, got "
                "Q="+str(pb.Q))
        if pb.metal: raise NotImplementedError("the convolution apply "
                "needs the same band window at every k-point, which a "
                "metallic pair basis does not have")
        g = pb.geometry
        self.pairs = pb
        self.kernel = kernel
        self.npair = pb.npair
        self.diag = np.array(pb.dE,dtype=np.float64)
        nk = len(pb.kpoints)
        nv,nc = len(pb.vbands),len(pb.cbands)
        norb = pb.el.shape[1]
        self.shape = (nk,nv,nc)
        self.nks = _mesh_shape(pb.kpoints,g.dimensionality)
        Cc = pb.el.reshape(nk,nv,nc,norb)[:,0] # (nk,nc,norb)
        Cv = pb.ho.reshape(nk,nv,nc,norb)[:,:,0] # (nk,nv,norb)
        # the transposes the two numba kernels read contiguously
        self.CcT = np.ascontiguousarray(np.swapaxes(Cc,1,2)) # (nk,norb,nc)
        self.CvcT = np.ascontiguousarray(np.swapaxes(np.conj(Cv),1,2))
        self.Ccc = np.ascontiguousarray(np.conj(Cc)) # (nk,nc,norb)
        self.Cv = np.ascontiguousarray(Cv) # (nk,nv,norb)
        # the orbital channels (a,b) that are needed, the ones where the
        # direct W does not vanish first, then the orbital diagonal for
        # the exchange term; only these are gathered, transformed and
        # scattered back
        chans = np.zeros(0,dtype=np.int64)
        self.direct = None
        if direct:
            # W at every mesh difference k_i - k_0, so that the direct
            # term is a cyclic convolution
            Wq = interaction_at_qs(W,g,pb.kpoints-pb.kpoints[0]) # (nk,norb,norb)
            Wq = Wq.reshape(nk,norb*norb)
            chans = np.nonzero(np.max(np.abs(Wq),axis=0)>1e-12)[0]
            Wq = Wq[:,chans].reshape(self.nks+(1,-1)) # over the columns
            axes = tuple(range(len(self.nks)))
            # the direct term enters as -D, with the 1/N of the kernel
            if len(chans)>0: self.direct = -np.fft.fftn(Wq,axes=axes)/nk
        self.ndirect = len(chans)
        self.exchange = None
        if exchange:
            WQ = interaction_at_q(Wx,g,pb.Q)/nk
            if np.max(np.abs(WQ))>1e-12:
                self.exchange = WQ
                diag = np.arange(norb)*(norb+1)
                chans = np.concatenate([chans,diag[~np.isin(diag,chans)]])
                # where (a,a) sits among the channels
                self.idiag = np.array([np.nonzero(chans==d)[0][0]
                    for d in diag])
        self.ia,self.ib = chans//norb,chans%norb
        self.nterm = int(self.direct is not None) + int(
                self.exchange is not None)
        self._buffers = dict() # the orbital matrices, per number of columns
    def matvec(self,x):
        """Apply the block to a vector or to a set of column vectors"""
        x = np.asarray(x)
        vec = x.ndim==1
        X = x.reshape(self.npair,-1)
        # C-ordered, so that the scatter below adds into a view of it
        out = np.multiply(self.diag[:,None],X,order="C",
                dtype=np.complex128)
        if self.nterm==0: return out[:,0] if vec else out
        nk,nv,nc = self.shape
        nch = len(self.ia)
        ncol = X.shape[1]
        if ncol not in self._buffers: # reused, a fresh one page-faults;
            # only the last column count is kept, so that the Lanczos
            # vector, the recursion block and to_dense do not all stay
            self._buffers = {ncol:(np.empty((nk,ncol,nch),
                dtype=np.complex128),np.empty((nk,ncol,nch),
                dtype=np.complex128))}
        Z,M = self._buffers[ncol]
        parallel.set_num_threads() # honor parallel.py's thread configuration
        Xk = np.ascontiguousarray(X,dtype=np.complex128).reshape(nk,nv,nc,ncol)
        # the pairs as orbital matrices, Z[k,n,s] = Phi_k[a_s,b_s]
        _gather(self.CcT,self.CvcT,Xk,self.ia,self.ib,Z)
        nd = self.ndirect
        if self.direct is not None:
            axes = tuple(range(len(self.nks)))
            Zr = Z.reshape(self.nks+(ncol,nch))[...,:nd]
            w = numba.get_num_threads()
            Y = scipy.fft.fftn(Zr,axes=axes,workers=w)
            Y *= self.direct
            Y = scipy.fft.ifftn(Y,axes=axes,workers=w,overwrite_x=True)
            M[:,:,:nd] = Y.reshape(nk,ncol,nd)
        if self.exchange is not None:
            rho = np.sum(Z[:,:,self.idiag],axis=0) # (ncol,norb), the mesh
            M[:,:,nd:] = 0. # the diagonal channels the direct term lacks
            M[:,:,self.idiag] += (rho@self.exchange.T)[None]
        # back onto the pairs, <C_c(k)|M(k)|C_v(k)>, added to the diagonal
        _scatter(self.Ccc,self.Cv,M,self.ia,self.ib,out.reshape(nk,nv,nc,ncol))
        return out[:,0] if vec else out
    def to_dense(self):
        """The dense block, for tests and small meshes only"""
        return self.matvec(np.eye(self.npair,dtype=np.complex128))


def _mesh_shape(kpoints,dim):
    """The shape of the regular mesh the k-points come from, checking that
    point i1*n+i2 sits at k_0 + (i1,i2)/n, which the cyclic convolution
    needs"""
    nk = len(kpoints)
    n = int(round(nk**(1./dim)))
    shape = (n,)*dim
    ok = n**dim==nk # checked first, unravel_index needs it
    if ok:
        idx = np.array(np.unravel_index(np.arange(nk),shape)).T # (nk,dim)
        dk = (kpoints[:,:dim] - kpoints[0,:dim]) - idx/n
        ok = np.max(np.abs(dk-np.round(dk)))<=1e-8
    if not ok:
        raise ValueError("the convolution apply needs the k-points of a "
                "regular n^%d mesh in the order of geometry.get_kmesh"%dim)
    return shape


@njit(parallel=True,cache=True)
def _gather(CcT,CvcT,X,ia,ib,Z):
    """Z[k,n,s] = sum_vc Cc[k,c,a] X[k,v,c,n] conj(Cv[k,v,b]) for the
    channels s = (a,b) = (ia[s],ib[s]), with the transposes
    CcT[k,a,c] = Cc[k,c,a] and CvcT[k,b,v] = conj(Cv[k,v,b])"""
    nk,nv,nc,ncol = X.shape
    norb = CcT.shape[1]
    for k in prange(nk): # loop over the mesh, in parallel
        T = np.empty((norb,nv),dtype=np.complex128)
        for n in range(ncol):
            for a in range(norb): # the conduction index first
                for v in range(nv):
                    acc = 0j
                    for c in range(nc): acc += X[k,v,c,n]*CcT[k,a,c]
                    T[a,v] = acc
            for s in range(len(ia)): # then the valence one
                a,b = ia[s],ib[s]
                acc = 0j
                for v in range(nv): acc += T[a,v]*CvcT[k,b,v]
                Z[k,n,s] = acc


@njit(parallel=True,cache=True)
def _scatter(Ccc,Cv,M,ia,ib,out):
    """out[k,v,c,n] += sum_s conj(Cc[k,c,a]) M[k,n,s] Cv[k,v,b] over the
    channels s = (a,b) = (ia[s],ib[s]), with Ccc = conj(Cc)"""
    nk,ncol,nch = M.shape
    norb = Cv.shape[2]
    nv,nc = out.shape[1],out.shape[2]
    for k in prange(nk): # loop over the mesh, in parallel
        T = np.empty((nv,norb),dtype=np.complex128)
        for n in range(ncol):
            T[:,:] = 0.
            for s in range(nch):
                a,b = ia[s],ib[s]
                m = M[k,n,s]
                for v in range(nv): T[v,a] += m*Cv[k,v,b]
            for v in range(nv):
                for c in range(nc):
                    acc = 0j
                    for a in range(norb): acc += Ccc[k,c,a]*T[v,a]
                    out[k,v,c,n] += acc
