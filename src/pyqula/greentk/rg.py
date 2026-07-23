import numpy as np
from .. import algebra
from numba import jit, prange
# sets numba.config.THREADING_LAYER = 'workqueue' (fork-safe) before any
# parallel=True numba function in the package gets compiled/run -- must be
# imported ahead of green_renormalization_jit_batch_core below
from .. import parallel

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




def green_renormalization_python(intra,inter,energy=0.0,nite=None,
                            info=False,delta=0.001,
                            **kwargs):
    """ Calculates bulk and surface Green function by a renormalization
    algorithm, as described in I. Phys. F: Met. Phys. 15 (1985) 851-858 """
    intra = algebra.todense(intra)
    inter = algebra.todense(inter)
    error = np.abs(delta)*1e-6 # overwrite error
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
    return g_bulk,g_surf

def green_renormalization_jit(intra,inter,energy=0.0,delta=1e-4,**kwargs):
    intra = algebra.todense(intra)*(1.0+0j)
    inter = algebra.todense(inter)*(1.0+0j)
    # same convergence criterion as green_renormalization_python, so this
    # path only changes speed (compiled loop), never the numerical result
    nite = max(int(100/np.abs(delta)),100000) # maximum number of iterations
    error = np.abs(delta)*1e-6
    energyz = energy + 1j*delta
    e = np.array(np.identity(intra.shape[0]),dtype=np.complex128) * energyz
    return green_renormalization_jit_core(intra,inter,e,nite,
                                                error)



## this is an optimized version
@jit(nopython=True)
def green_renormalization_jit_core(intra, inter, e, nite, error):
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
        if np.max(np.abs(alpha)) < error and np.max(np.abs(beta)) < error:
            break
        if ite >= nite:
            break
    # Compute Green's functions using solve (avoids explicit inverse)
    I = np.eye(n, dtype=epsilon.dtype)
    g_surf = np.linalg.solve(e - epsilon_s, I)
    g_bulk = np.linalg.solve(e - epsilon, I)

    return g_bulk, g_surf


def green_renormalization_jit_batch(intra,inter,energies,delta=1e-4,**kwargs):
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
    # same convergence criterion as green_renormalization_python/_jit, so
    # this path only changes speed, never the numerical result
    nite = max(int(100/np.abs(delta)),100000) # maximum number of iterations
    error = np.abs(delta)*1e-6
    return green_renormalization_jit_batch_core(intra,inter,energies,delta,
                                                 nite,error)


@jit(nopython=True,parallel=True,cache=True)
def green_renormalization_jit_batch_core(intra,inter,energies,delta,nite,error):
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
            if np.max(np.abs(alpha)) < error and np.max(np.abs(beta)) < error:
                break
            if ite >= nite:
                break
        I = np.eye(n, dtype=epsilon.dtype)
        g_surf[k] = np.linalg.solve(e - epsilon_s, I)
        g_bulk[k] = np.linalg.solve(e - epsilon, I)
    return g_bulk, g_surf


