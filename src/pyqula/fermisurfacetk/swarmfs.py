import numpy as np
import jax.numpy as jnp
import jax
import jax.numpy.linalg as jlg

def fermi_surface(H,nk=100,reciprocal=True,nrep=1):
    """Return the Fermi surface with swarm optimization"""
    hk = H.get_hk_gen(use_jax=True) # get the generator
    def fun(k):
        """Function to optimize"""
        h = hk(k) # get matrix
#        iden = jnp.identity(h.shape[0])
        delta= 1e-4
#        gf = 1./jnp.trace(jlg.inv(h-iden*1j*delta)).imag
#        return gf
        es = jlg.eigvalsh(h) # diagonalize
        return jnp.log(1./jnp.sum(1./(es**2+delta)).real)
#        return jnp.abs(jlg.slogdet(h)[1].real) # determinant
    # transform
    dim = H.dimensionality
    if reciprocal: 
        R = H.geometry.get_k2K()[0:dim,0:dim] # get function
        R = jlg.inv(np.array(R))
    else:  R = np.identity(dim)
    #############
    gradfun = jax.grad(fun) # gradient
    hessfun = jax.hessian(fun) # hessian
#    hessfun = None
#    gradfun = None
    from scipy.optimize import minimize
    tol = 1e-4 # tolerance
    k0 = np.random.random(H.dimensionality) - 0.5 # starting point
    # kmesh 
    nk2 = int(np.sqrt(nk))
    from .. import klist
    ks = klist.kmesh(H.dimensionality,nk=nk2)[:,0:2]
    k0 = ks[0]
    nk = len(ks)
    #####
    out = np.zeros((nk,k0.shape[0]),dtype=np.float64) # zeros
    ik = 0 # start
    while True: # infinite loop
        result = minimize(fun,k0,tol=tol,jac=gradfun,hess=None,
                          method="Newton-CG")
        print(np.exp(fun(result.x)),result.x)
        if np.exp(fun(result.x))<tol: # if found a point
            o = result.x%1. # output
            out[ik] = o # store
            ik += 1 # next one
        if np.random.random()<0.5 or ik<2: # random starting point
            k0 = np.random.random(H.dimensionality) - 0.5 # starting point
        else: # follow the hessian
        #    k0 = select_unexplored_kpoint(out[0:ik,:]) # unexplored kpoint
            k0 = generate_another_zero(result.x,hessfun)
        #    k0 = select_unexplored_kpoint(out[0:ik,:]) # unexplored kpoint
        #    k0 = k0 + 0.05*(np.random.random(H.dimensionality) - 0.5)
        if ik>=nk: break # stop loop
    out2 = []
    for o in out:
            for j1 in range(nrep): # loop
                for j2 in range(nrep): # loop
                    op = o + np.array([j1,j2])
                    out2.append(R@op) # store
    return np.array(out2) # return kpoints

from numba import jit


import scipy.linalg as lg

def generate_another_zero(k,hess,dk=0.1):
    """Given a kpoint where the function is zero and a Hessian,
    generate another kpoint where the function is zero
    following the zero direction of the hessian"""
    m = hess(k) # generate hessian matrix
    (eigs,vs) = lg.eigh(m) # eigenvalues and eigenvectors
    print("Hess eig",eigs)
    ind = np.argmin(np.abs(eigs)) # minimum eigenvalue index
    v = vs[:,ind] # direction of the zero direction
    r = np.random.random()-0.5
    return k + r*dk*v # return new suggested kpoint







@jit(nopython=True)
def select_unexplored_kpoint(ks):
    """Given a list of kpoints, select an intermediate kpoint
    close to the most unexplored kpoint"""
    nk = len(ks) # number of kpoints
    dmax = 100 # maximum distance
    zs = np.exp(1j*ks) # in complex plane
    dsmin = np.zeros(nk,dtype=np.float64) # minimum distances
    jsmin = np.zeros(nk,dtype=np.int64) # minimum indexes
    for i in range(nk): # loop over kpoints
        zi = zs[i] # in complex plane
        dzmin = 10 # minimum distance
        for j in range(nk): # loop over remaning kpoints
            zj = zs[j] # in complex plane
            dzj = np.mean(np.abs(zi-zj)) # distance in complex plane
            if dzj<dzmin and i!=j: 
                jmin = j # store 
                dzmin = dzj # store
        jsmin[i] = jmin # store index
        dsmin[i] = dzmin # store distance
    ialone = np.argmax(dsmin) # index of the most isolated kpoint
    return ks[ialone]
#    print(np.max(dsmin))
    dk = ks[ialone]-ks[jsmin[ialone]] # intermediate kpoint
    dk = dk/np.sqrt(np.sum(dk**2)) # normalize
    ko = ks[ialone] + 0.05*dk # unexplored point
    return ko # return new kpoint



