"""Matrix-free BSE solver: the few lowest excitons without the matrix.

solve.check_memory refuses a calculation whose dense BSE matrix does not
fit, and npair = nk*nv*nc grows fast in exactly the knob a user turns
first. This module removes that wall: bsetk/factorize.py writes the
resonant block as a diagonal plus a fixed number of rank-one terms, so the
matrix-vector product costs O(nterm*npair) and needs O(nterm*npair)
memory. A block eigensolver on top of that returns the lowest excitons at
meshes where the dense matrix would be hundreds of gigabytes.

This is exact, not an approximation: the operator applied is the same
resonant block build_blocks assembles, to machine precision. The only
approximation is the eigensolver's own convergence.

TAMM-DANCOFF ONLY. The full (non-Tamm-Dancoff) problem diagonalizes
[[A,B],[-B^dag,-conj(Abar)]], which is not Hermitian; solve.py handles it
by Cholesky-factorizing S@H, and that needs the dense matrix. Doing it
matrix-free would need a Lanczos run in the S@H inner product, which is
not built here. Since TDA converges to the full answer at weak coupling
(tests/bse/test_bse_physics.py) and large-scale BSE codes use it for the
same reason, the dense solver stays the route for the full problem at
small meshes and this one is the route for large ones.

Cost, honestly: this is O(nk) diagonalizations of h plus O(nterm*npair)
work per matrix-vector product, so it is AT LEAST linear in the mesh
rather than logarithmic (0.07 / 0.09 / 0.24 s at nk = 256 / 1024 /
4096 on the spinless chain of tests/bse/test_bse_iterative.py). It removes the memory wall, not the linear scaling;
bsetk/qtt.py is what addresses the latter. In exchange it needs no gauge
fixing, no tensor-train tolerance and no extra dependency, so it is also
the reference that harder sibling is checked against.

THE EIGENSOLVER HAS TWO TRAPS IN IT, and the fast option loses to both.

ARPACK is the obvious first choice: it is fast and its eigenvalues are
exact. But a
single-start-vector Lanczos CANNOT resolve
eigenvalue MULTIPLICITY: the Krylov space it builds contains each
distinct eigenvalue once, so a degenerate level comes back once and the
remaining slots are filled from higher up, with rounding noise deciding
how much of the multiplet survives. It is therefore nondeterministic --
on the spinful ionic chain at nk=16, whose lowest exciton is four-fold
degenerate, repeated runs returned either the correct 1.65611456 x4 or
1.65611456 x3 followed by 1.88540220, a 0.16 error wearing a converged
face. Exciton spectra are degenerate as a rule, not as an exception (a
spinful model with no spin-orbit coupling makes every transition a
four-fold multiplet, and the singlet/triplet structure IS the physics),
so this is not a corner case that can be documented away.

LOBPCG is a block method and resolves multiplicity by construction, but
it is slow from a cold start and its obvious preconditioner is a trap.
The BSE block is strongly diagonally dominant -- the diagonal is the
O(gap) transition energy against an O(1/N) kernel -- which makes a Jacobi
preconditioner 1/(dE - min dE) look ideal. It is not, and it fails worse
the finer the mesh, i.e. exactly where this module is meant to be used.
Measured against the exact answer 1.5023327683:

  nk      unshifted Jacobi   shifted Jacobi   no preconditioner
  1024    3.8e-05            3.8e-12          3.8e-12
  4096    2.9e-01            3.8e-12          3.8e-12

Refining the mesh packs the transition energies near the band-edge
minimum into an ever denser cluster -- the spread scales like 1/nk^2 --
so 1/(dE - min dE) becomes enormous and nearly constant across hundreds
of states, preconditioning nothing and destroying the conditioning.
Shifting the denominator by a fixed fraction of the diagonal's own spread
removes the blow-up and restores exactness at every mesh size tried.

Seeding LOBPCG from ARPACK looks like the best of both and is not: the
seeded block inherits ARPACK's inability to span a degenerate eigenspace,
so on the four-fold degenerate case one run in three still came back 0.23
off while the others were exact to 4e-15. Nondeterminism is exactly what
a reference implementation must not have.

So what is implemented is a shifted-Jacobi-preconditioned LOBPCG from a
DETERMINISTIC starting block -- unit vectors on the smallest diagonal
entries, a few columns wider than neig and ending, where it can, at
the end of a degenerate multiplet of the transition energies (see
_block_size), stopped on the residuals of the wanted excitons only. Exact to
4e-15 on the degenerate case and 3.8e-12 at every mesh size tried, run to
run, which is the point of a reference implementation: the same answer
every time. Stopping on the whole block instead ran every call to maxiter,
and cost 0.34 / 0.34 / 2.68 s at nk = 256 / 1024 / 4096.
"""
import numpy as np
import scipy.sparse.linalg as sla

from .factorize import KernelFactorization


def solve_iterative(pb,W,Wx=None,kernel="full",neig=4,maxiter=800,
        tol=1e-12,shift=0.1):
    """Return (energies,amplitudes) for the neig lowest excitons of the
    resonant (Tamm-Dancoff) BSE block.

    shift sets the preconditioner's regularization as a fraction of the
    diagonal's own spread; it is the difference between a correct answer
    and a plausible wrong one, and this module's docstring has the
    measurements. tol and maxiter are LOBPCG's.

    amplitudes come back in the same (nexciton,npair) layout
    BSE.amplitudes uses, so downstream code does not have to care which
    solver ran."""
    f = KernelFactorization(pb,W,Wx=Wx,kernel=kernel,block="A")
    if neig<1:
        raise ValueError("neig must be at least 1")
    npair = pb.npair
    if npair<=max(4*neig,32):
        # too small for an iterative solver to be either faster or more
        # reliable, and the dense block is a few tens of kB at this size
        import scipy.linalg as lg
        es,ws = lg.eigh(f.to_dense())
        n = min(neig,npair)
        return es[0:n],np.array(ws.T[0:n],dtype=np.complex128)
    op = as_linear_operator(f)
    # the block size and the starting block follow the multiplets of the
    # transition energies dE, which the kernel diagonal splits at the 1/N
    # level, so that seeding them from f.diagonal() never grew the block
    # past a multiplet (3.7 s against 0.4 s on the spinful honeycomb
    # lattice with an extended interaction); the preconditioner keeps the
    # whole diagonal
    de = np.array(f.diag,dtype=np.float64)
    X = _starting_block(de,_block_size(de,neig,npair),npair)
    M = _preconditioner(f.diagonal().real,npair,shift)
    return _lobpcg_lowest(op,f.matvec,X,M,neig,tol,maxiter)


def _lobpcg_lowest(op,matvec,X,M,neig,tol,maxiter,chunk=20):
    """The neig lowest eigenpairs from a block LOBPCG, stopped on the
    residuals of those neig only.

    scipy's tol applies to every column of the block, and the columns past
    neig, which are there to span a degenerate cluster, converge last, so
    a single call always ran to maxiter (800 iterations where 30 gave the
    same energies to 1e-15 on the spinful ionic chain). The block is
    therefore run in chunks, each restarted from the Ritz vectors of the
    previous one, until the residuals |A x - E x| of the wanted neig fall
    below tol. An eigenvalue error is quadratic in the residual, so the
    energies are still good to about tol when the residual only reaches
    sqrt(tol); above that the result is returned with a warning rather
    than as if it had converged"""
    import warnings
    done = 0
    while True:
        it = min(chunk,maxiter-done)
        with warnings.catch_warnings():
            # LOBPCG warns whenever it stops short of the requested
            # residual on any column of the block, which the check below
            # replaces with one on the wanted columns
            warnings.simplefilter("ignore")
            es,ws = sla.lobpcg(op,X,M=M,tol=tol,maxiter=it,largest=False)
        done += it
        order = np.argsort(es.real)[0:neig]
        e,w = es.real[order],ws[:,order]
        res = np.linalg.norm(matvec(w) - w*e[None,:],axis=0)
        res = np.max(res/np.linalg.norm(w,axis=0))
        if res<=tol or done>=maxiter: break
        X = ws
    if res>np.sqrt(tol): warnings.warn("LOBPCG stopped after %d iterations "
            "with a residual %.1e on the %d lowest excitons, above %.1e: "
            "the energies are only accurate to about the residual squared "
            "over the level spacing. Raise maxiter"%(done,res,neig,
                np.sqrt(tol)),stacklevel=3)
    return (np.array(e,dtype=np.float64),
            np.array(w.T,dtype=np.complex128))


def _starting_block(d,nblock,npair):
    """A deterministic starting block: unit vectors on the smallest
    diagonal entries, plus a fixed-seed random perturbation to break exact
    ties.

    Deterministic on purpose. The alternative -- seeding from ARPACK, which
    is cheap and lands close -- was tried and rejected: it inherits
    ARPACK's own inability to span a degenerate eigenspace, so one run in
    three came back 0.23 off on the four-fold degenerate case while the
    others were exact to 4e-15. A solver whose answer depends on a random
    start vector is not a reference anything else can be checked against,
    and being a reference is this module's main job."""
    X = np.zeros((npair,nblock),dtype=np.complex128)
    for j,i in enumerate(np.argsort(d)[0:nblock]): X[i,j] = 1.
    rng = np.random.default_rng(0)
    return X + 1e-3*rng.normal(size=(npair,nblock))


def _preconditioner(d,npair,shift):
    """Jacobi preconditioner, regularized by a fraction of the diagonal's
    spread. Without the shift this is actively harmful; see the module
    docstring."""
    spread = float(np.max(d)-np.min(d))
    inv = 1./np.maximum(d-np.min(d)+shift*max(spread,1e-6),1e-12)
    def apply(x):
        return inv[:,None]*x.reshape(npair,-1) if np.ndim(x)>1 else inv*x
    return sla.LinearOperator((npair,npair),matvec=apply,
            dtype=np.complex128)


def _block_size(d,neig,npair,tol=1e-8):
    """How wide the eigensolver's block should be.

    At least neig, and preferably ending where a degenerate multiplet of
    the diagonal ends. The interaction is a small perturbation on the
    transition energies, so a multiplet of dE is a near-multiplet of the
    block, and the columns past neig converge fastest when the block
    holds a whole one. The target is neig + max(8,neig) columns; the
    block ends at the last multiplet boundary between neig and the
    target, or, if there is none, at the first one past the target
    within as many columns again, or at the target itself.

    Searching down first is what keeps the block small on a 2D lattice,
    where a multiplet of dE is a star of k-points times the spin, 48
    states on the ring around K and K' of the spinful honeycomb lattice
    at nk=48. Growing up to the end of that ring made the block 56 wide
    for neig=4, and 10 s against 1.2 s for the block of 8 that ends at
    the boundary below, with the same energies to twelve digits. Cutting
    a multiplet costs speed, not exactness: blocks of 12 and 16 there
    give the same energies too, which is why the target is kept when no
    boundary is near."""
    ds = np.sort(d)
    last = npair-1 # LOBPCG wants a block narrower than the matrix
    k0 = min(neig+max(8,neig),last)
    def boundary(k): # is there a gap between the k-th and (k+1)-th entry?
        return abs(ds[k]-ds[k-1])>=tol*max(1.,abs(ds[k-1]))
    for k in range(k0,neig-1,-1): # the last boundary below the target
        if boundary(k): return k
    for k in range(k0+1,min(2*k0-neig,last)+1): # the first one above
        if boundary(k): return k
    return k0


def as_linear_operator(f):
    """Wrap a KernelFactorization as a scipy LinearOperator.

    The operator is Hermitian, so rmatvec is matvec; scipy needs it
    spelled out anyway."""
    n = f.npair
    return sla.LinearOperator((n,n),matvec=f.matvec,rmatvec=f.matvec,
            matmat=f.matvec,rmatmat=f.matvec,dtype=np.complex128)
