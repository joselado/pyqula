import numpy as np
import threading
from .. import algebra
from numba import jit, prange
# sets numba.config.THREADING_LAYER = 'workqueue' (fork-safe) before any
# parallel=True numba function in the package gets compiled/run -- must be
# imported ahead of green_renormalization_jit_batch_core below
from .. import parallel

# numba's 'workqueue' threading layer -- which parallel.py selects for
# fork-safety -- is NOT threadsafe: entering a parallel=True kernel from two
# Python threads at once aborts the interpreter ("Fatal Python error:
# Aborted" inside numba/np/ufunc/workqueue).
#
# That is reachable in practice: keldyshtk/current.py's build_selfenergy_aaa
# builds the two leads' AAA interpolants in a ThreadPoolExecutor (4a086f5),
# and both threads land in green_renormalization_jit_batch_core below. It is
# a race, so it fires intermittently -- observed killing a `pytest
# tests/keldysh` run partway through, with two [ThreadPoolExecu] threads in
# the traceback both inside this kernel.
#
# Guard the kernel itself rather than that one call site: any future
# multi-threaded caller is then safe by construction. The lock costs nothing
# in the (usual) single-threaded case, and concurrent callers still overlap
# their non-numba work -- the LAPACK/SVD parts of an AAA build release the
# GIL and are unaffected by this lock.
_batch_lock = threading.Lock()

use_numba = False # default backend for green_renormalization

def green_renormalization(intra,inter,numba=None,**kwargs):
    """Dispatch to the numba-jitted or pure-Python Sancho-Rubio iteration.
    `numba` overrides the module-level default (`greentk.rg.use_numba`) for
    this call only -- callers that need the numba path for a hot loop (e.g.
    keldyshtk/current.py, which recomputes lead selfenergies many thousands
    of times) can opt in without changing the default used by every other
    DOS/LDOS/transport call in the library."""
    if numba is None: numba = use_numba
    if numba:
      return green_renormalization_jit(intra,inter,**kwargs)
    else:
      return green_renormalization_python(intra,inter,**kwargs)




# Tolerance on the Dyson residual below which the decimation's answer is
# accepted, see surface_dyson_residual. A correct decimation gives 1e-12
# or better at ordinary broadenings and ~1e-4 at the very smallest ones,
# while the failure mode below is off by O(1) or worse -- so this
# separates them by many orders of magnitude and never fires on a good
# result.
dyson_tolerance = 1e-3


def surface_dyson_residual(gs,intra,inter,e):
    """Residual of the equation the surface Green's function must satisfy,

        g_s = (e - intra - inter g_s inter^dag)^(-1)

    measured as max|(e - intra - inter g_s inter^dag) g_s - 1|.

    This is a cheap, independent check on the Sancho-Rubio decimation
    below, which is not unconditionally accurate: at an energy that sits
    exactly on an onsite level of the lead (E=0 for the usual
    particle-hole symmetric lead, with intra=0) the very first decimation
    step has to invert e - intra = i*delta, so for a tiny delta the whole
    recursion runs on catastrophically cancelled numbers. It still
    "converges" -- alpha and beta do fall below the threshold -- but to a
    wrong fixed point: at E=0, delta=1e-12, a semi-infinite chain came out
    with a surface Green's function of -8932j where the exact value is
    -1j. Since transporttk/smatrix.py evaluates every S-matrix at
    delta=1e-12, that silently made the zero-bias conductance of a
    junction ~0 instead of the Landauer/BTK value."""
    n = intra.shape[0]
    m = e - intra - inter@gs@algebra.dagger(inter)
    return np.max(np.abs(m@gs - np.identity(n,dtype=np.complex128)))


def surface_dyson_residual_batch(g_surf,intra,inter,energies,delta):
    """surface_dyson_residual for a whole batch of energies at once.

    One residual per energy, computed with broadcast matrix products
    instead of a Python loop -- the loop version cost as much as the
    batched decimation itself, which would have made this check the
    dominant term in the Keldysh sideband sweeps that use it."""
    n = intra.shape[0]
    idag = algebra.dagger(inter)
    sig = inter@g_surf@idag # (nE,n,n), broadcast over the batch
    ez = np.asarray(energies) + 1j*delta # complex energies
    m = -intra[None,:,:] - sig
    m[:,np.arange(n),np.arange(n)] += ez[:,None] # add (E+i*delta)*identity
    r = m@g_surf - np.identity(n,dtype=np.complex128)[None,:,:]
    return np.max(np.abs(r),axis=(1,2))


def surface_green_dyson(intra,inter,e,mix=0.5,nite=20000,tol=1e-14):
    """Surface Green's function from a damped fixed point iteration of the
    Dyson equation g = (e - intra - inter g inter^dag)^(-1), starting from
    g = 0.

    Much slower to converge than the decimation (tens to a few hundred
    iterations rather than ~20, since it adds one cell at a time instead
    of doubling), but every iteration is well conditioned -- it never
    forms the huge intermediate quantities that break the decimation at a
    degenerate energy -- so it is used as the fallback there.

    `tol` is measured relative to the size of the iterate: where the true
    surface Green's function is of order 1/delta an absolute 1e-14 is
    simply unreachable, and the iteration would grind through every one
    of `nite` steps for nothing. The iteration starts at g=0 and grows,
    so at such an energy `e - intra - inter g inter^dag` can itself
    collapse to a numerically singular matrix; that is not an error to
    report here -- the current iterate is returned and the caller's Dyson
    check decides whether anything usable came out."""
    n = intra.shape[0]
    g = np.zeros((n,n),dtype=np.complex128)
    iden = np.identity(n,dtype=np.complex128)
    for i in range(nite):
        try:
            gn = np.linalg.solve(e - intra - inter@g@algebra.dagger(inter),
                                 iden)
        except np.linalg.LinAlgError: return g # no meaningful step left
        gn = mix*gn + (1.-mix)*g # damping
        scale = max(np.max(np.abs(gn)),1.) # the iterate can be huge
        if np.max(np.abs(gn-g))<tol*scale: return gn
        g = gn
    return g


def _fix_green_renormalization(g_bulk,g_surf,intra,inter,e):
    """Accept the decimation's Green's functions, or replace them by the
    fixed-point ones if they fail their own Dyson equation.

    The fixed point is used only if it actually satisfies the equation;
    the bulk Green's function is then rebuilt from it, which needs the
    surface Green's function of *both* semi-infinite halves,
    g_b = (e - intra - inter g_s inter^dag - inter^dag g_s' inter)^(-1).

    If neither of the two satisfies the equation there is no answer to
    hand back, and this raises. That happens when the lead has a state
    essentially at the evaluated energy: the true surface Green's
    function then grows like 1/delta and has to come out of inverting a
    matrix assembled by cancelling numbers of that same size, which
    double precision cannot carry -- a multi-orbital lead with intra=0 at
    E=0 and delta=1e-12 is the standard example. Returning the less-bad
    of two wrong Green's functions is exactly the silent failure this
    whole residual check exists to stop, so say so instead."""
    res = surface_dyson_residual(g_surf,intra,inter,e)
    if res<dyson_tolerance: return g_bulk,g_surf # the decimation is fine
    gsr = surface_green_dyson(intra,inter,e) # this side
    res2 = surface_dyson_residual(gsr,intra,inter,e)
    if not res2<dyson_tolerance: # neither of the two solves the equation
        raise ValueError("the surface Green's function of this lead does "
                "not satisfy its own Dyson equation at energy %g and "
                "delta %g (best residual %.2e, required below %.1e). The "
                "lead has a state essentially at that energy, where the "
                "surface Green's function grows like 1/delta and double "
                "precision cannot resolve it; use a larger delta, or "
                "evaluate away from that energy."
                %(e[0,0].real,e[0,0].imag,min(res,res2),dyson_tolerance))
    dag = algebra.dagger
    gsl = surface_green_dyson(intra,dag(inter),e) # the opposite side
    n = intra.shape[0]
    iden = np.identity(n,dtype=np.complex128)
    gb = np.linalg.solve(e - intra - inter@gsr@dag(inter)
                            - dag(inter)@gsl@inter, iden)
    return gb,gsr


def green_renormalization_python(intra,inter,energy=0.0,nite=None,
                            info=False,delta=0.001,error=None,
                            **kwargs):
    """ Calculates bulk and surface Green function by a renormalization
    algorithm, as described in I. Phys. F: Met. Phys. 15 (1985) 851-858

    `error` is the threshold on the decimated couplings below which the
    iteration stops, defaulting to |delta|*1e-6; it used to be accepted
    from callers (heterostructures.calculate_surface_green passes one) and
    then silently overwritten. `nite`, if given, fixes the number of
    iterations instead, i.e. asks for a *truncated* decimation -- the
    Dyson check at the end is then skipped, since a truncated result is
    meant to be unconverged."""
    intra = algebra.todense(intra)
    inter = algebra.todense(inter)
    if error is None: error = np.abs(delta)*1e-6 # default threshold
    n = intra.shape[0]
    e = np.identity(n,dtype=np.complex128) * (energy + 1j*delta)
    ite = 0
    alpha = inter.copy()
    beta = algebra.dagger(inter).copy()
    epsilon = intra.copy()
    epsilon_s = intra.copy()
    # Both updates below only ever need alpha@einv and beta@einv acting on
    # {alpha,beta}, so solve the two right-hand-sides at once with a single
    # LU factorization instead of forming the explicit inverse each iteration.
    rhs = np.empty((n,2*n),dtype=np.complex128)
    while True: # implementation of Eq 11
      rhs[:,:n] = beta
      rhs[:,n:] = alpha
      sol = np.linalg.solve(e - epsilon, rhs) # [einv@beta | einv@alpha]
      alpha_sol = alpha @ sol # [alpha@einv@beta | alpha@einv@alpha]
      beta_sol = beta @ sol   # [beta@einv@beta  | beta@einv@alpha]
      epsilon_s = epsilon_s + alpha_sol[:,:n]
      epsilon = epsilon + alpha_sol[:,:n] + beta_sol[:,n:]
      alpha = alpha_sol[:,n:]  # new alpha
      beta = beta_sol[:,:n]  # new beta
      ite += 1
      # stop conditions
      if not nite is None:
        if ite > nite:  break
      else:
        if np.max(np.abs(alpha))<error and np.max(np.abs(beta))<error: break
    if info:
      print("Converged in ",ite,"iterations")
    identity = np.identity(n,dtype=np.complex128)
    g_surf = np.linalg.solve(e - epsilon_s, identity) # surface green function
    g_bulk = np.linalg.solve(e - epsilon, identity)  # bulk green function
    if nite is not None: return g_bulk,g_surf # deliberately truncated
    # the decimation is not unconditionally accurate, check it
    return _fix_green_renormalization(g_bulk,g_surf,intra,inter,e)

def green_renormalization_jit(intra,inter,energy=0.0,delta=1e-4,
                              nite=None,error=None,**kwargs):
    """Numba-compiled twin of green_renormalization_python. `nite` and
    `error` mean exactly what they mean there -- a truncated decimation and
    a convergence threshold -- and used to be recomputed here instead of
    being honoured, so the two backends answered different questions."""
    intra = algebra.todense(intra)*(1.0+0j)
    inter = algebra.todense(inter)*(1.0+0j)
    truncate = nite is not None # caller asked for a truncated decimation
    # same convergence criterion as green_renormalization_python, so this
    # path only changes speed (compiled loop), never the numerical result
    if nite is None: nite = max(int(100/np.abs(delta)),100000) # max iterations
    if error is None: error = np.abs(delta)*1e-6 # default threshold
    energyz = energy + 1j*delta
    e = np.array(np.identity(intra.shape[0]),dtype=np.complex128) * energyz
    g_bulk,g_surf = green_renormalization_jit_core(intra,inter,e,nite,error,
                                                   truncate)
    if truncate: return g_bulk,g_surf # deliberately unconverged, do not fix
    # same validity check as the pure Python path, so the two agree
    return _fix_green_renormalization(g_bulk,g_surf,intra,inter,e)



## this is an optimized version
@jit(nopython=True)
def green_renormalization_jit_core(intra, inter, e, nite, error, truncate):
    ite = 0
    # Force C‑contiguity from the start
    alpha = np.ascontiguousarray(inter * 1.0)
    beta = np.ascontiguousarray(np.conjugate(inter).T * 1.0)
    epsilon = np.ascontiguousarray(intra * 1.0)
    epsilon_s = np.ascontiguousarray(intra * 1.0)

    n = alpha.shape[0]
    # Pre‑allocate buffers for the stacked RHS and solution
    RHS = np.empty((n, 2 * n), dtype=alpha.dtype)
    ZY = np.empty((n, 2 * n), dtype=alpha.dtype)
    while True:
        A = e - epsilon
        # Solve A @ [Z | Y] = [beta | alpha]
        #  → Z = A⁻¹ @ beta, Y = A⁻¹ @ alpha
        RHS[:, :n] = beta
        RHS[:, n:] = alpha
        sol = np.linalg.solve(A, RHS)          # single LU decomposition
        ZY[:, :n] = sol[:, :n]                 # Z
        ZY[:, n:] = sol[:, n:]                 # Y
        # Two matrix multiplications instead of four
        alphaZY = alpha @ ZY   # [αZ | αY]
        betaZY = beta @ ZY     # [βZ | βY]
        # Extract the needed parts (slices are views)
        alphaZ = alphaZY[:, :n]
        alphaY = alphaZY[:, n:]
        betaZ = betaZY[:, :n]
        betaY = betaZY[:, n:]
        # Update surface and bulk self‑energies
        epsilon_s = epsilon_s + alphaZ
        epsilon = epsilon + alphaZ + betaY
        # New alpha and beta – copy to ensure C‑contiguity for next iteration
        alpha = alphaY.copy()
        beta = betaZ.copy()
        ite += 1
        if truncate: # fixed iteration count, no convergence test
            if ite > nite:
                break
        else:
            if np.max(np.abs(alpha)) < error and np.max(np.abs(beta)) < error:
                break
            if ite >= nite:
                break
    # Compute Green's functions using solve (avoids explicit inverse)
    I = np.eye(n, dtype=epsilon.dtype)
    g_surf = np.linalg.solve(e - epsilon_s, I)
    g_bulk = np.linalg.solve(e - epsilon, I)

    return g_bulk, g_surf


def green_renormalization_jit_batch(intra,inter,energies,delta=1e-4,
                                    nite=None,error=None,**kwargs):
    """Batched version of green_renormalization_jit: same lead (intra,
    inter fixed), many energies at once. The Sancho-Rubio iteration is
    completely independent across energies (only the starting `intra`,
    `inter` are shared), so this is an embarrassingly parallel batch,
    computed with a numba `prange` loop (see
    green_renormalization_jit_batch_core) instead of one Python-level call
    per energy. Useful when the same lead's selfenergy is needed at many
    energies at once, e.g. every Floquet sideband of a fixed quasienergy
    in keldyshtk/current.py -- it amortizes the numba/LAPACK call overhead
    across the whole batch and runs the sidebands over multiple threads."""
    intra = algebra.todense(intra)*(1.0+0j)
    inter = algebra.todense(inter)*(1.0+0j)
    energies = np.asarray(energies,dtype=np.float64)
    truncate = nite is not None # caller asked for a truncated decimation
    # same convergence criterion as green_renormalization_python/_jit, so
    # this path only changes speed, never the numerical result
    if nite is None: nite = max(int(100/np.abs(delta)),100000) # max iterations
    if error is None: error = np.abs(delta)*1e-6 # default threshold
    with _batch_lock: # workqueue is not threadsafe -- see _batch_lock above
        g_bulk,g_surf = green_renormalization_jit_batch_core(intra,inter,
                energies,delta,nite,error,truncate)
    if truncate: return g_bulk,g_surf # deliberately unconverged, do not fix
    # same validity check as the single-energy paths; the residuals for
    # the whole batch come from one set of broadcast matrix products, and
    # only the energies that actually fail go through the (rare, slow)
    # fixed-point fallback
    res = surface_dyson_residual_batch(g_surf,intra,inter,energies,delta)
    for k in np.where(res>=dyson_tolerance)[0]:
        e = np.identity(intra.shape[0],dtype=np.complex128)*(energies[k]+1j*delta)
        gb,gs = _fix_green_renormalization(g_bulk[k],g_surf[k],intra,inter,e)
        g_bulk[k],g_surf[k] = gb,gs
    return g_bulk,g_surf


@jit(nopython=True,parallel=True,cache=True)
def green_renormalization_jit_batch_core(intra,inter,energies,delta,nite,
                                         error,truncate):
    nE = energies.shape[0]
    n = intra.shape[0]
    g_bulk = np.empty((nE,n,n),dtype=np.complex128)
    g_surf = np.empty((nE,n,n),dtype=np.complex128)
    for k in prange(nE): # sidebands/energies are independent -> parallel
        e = np.eye(n,dtype=np.complex128)*(energies[k]+1j*delta)
        ite = 0
        alpha = np.ascontiguousarray(inter * 1.0)
        beta = np.ascontiguousarray(np.conjugate(inter).T * 1.0)
        epsilon = np.ascontiguousarray(intra * 1.0)
        epsilon_s = np.ascontiguousarray(intra * 1.0)
        RHS = np.empty((n, 2 * n), dtype=alpha.dtype)
        ZY = np.empty((n, 2 * n), dtype=alpha.dtype)
        while True:
            A = e - epsilon
            RHS[:, :n] = beta
            RHS[:, n:] = alpha
            sol = np.linalg.solve(A, RHS)
            ZY[:, :n] = sol[:, :n]
            ZY[:, n:] = sol[:, n:]
            alphaZY = alpha @ ZY
            betaZY = beta @ ZY
            alphaZ = alphaZY[:, :n]
            alphaY = alphaZY[:, n:]
            betaZ = betaZY[:, :n]
            betaY = betaZY[:, n:]
            epsilon_s = epsilon_s + alphaZ
            epsilon = epsilon + alphaZ + betaY
            alpha = alphaY.copy()
            beta = betaZ.copy()
            ite += 1
            if truncate: # fixed iteration count, no convergence test
                if ite > nite:
                    break
            else:
                if np.max(np.abs(alpha))<error and np.max(np.abs(beta))<error:
                    break
                if ite >= nite:
                    break
        I = np.eye(n, dtype=epsilon.dtype)
        g_surf[k] = np.linalg.solve(e - epsilon_s, I)
        g_bulk[k] = np.linalg.solve(e - epsilon, I)
    return g_bulk, g_surf


