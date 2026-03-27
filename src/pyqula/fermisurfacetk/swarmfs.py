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
    tol = 1e-4 # tolerance
    from scipy.optimize import minimize
    k0 = np.random.random(H.dimensionality) - 0.5 # starting point
    out = np.zeros((nk,k0.shape[0]),dtype=np.float64) # zeros found
    out2 = None # empty list
    ik = 0 # start
    while True: # infinite loop
        result = minimize(fun,k0,tol=None,jac=gradfun,hess=None,
                          method="Newton-CG",options={'xtol':tol})
        print(np.exp(fun(result.x)),result.x,ik)
        if np.exp(fun(result.x))<tol: # if found a point
            o = result.x%1. # output
            out[ik] = o # store
            ik += 1 # next one
        if np.random.random()<.2 or ik<2: # random starting point
            k0 = np.random.random(H.dimensionality) - 0.5 # starting point
        else: # follow the hessian
            k0,out2 = generate_unexplored_zero(out[0:ik,:],out2,hessfun)
            print("New ",k0)
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


def generate_unexplored_zero(ks1,ks2,hessfun,dk=1.,**kwargs):
    """Generate the new family of kpoints based on the Hessian"""
    rdk = lambda : dk*(np.random.random())
#    print(ks1,ks2)
    if ks2 is None:  k = ks1[len(ks1)-1] # last obtained kpoint
    else:
        k = select_unexplored_kpoint(ks2,ks1) # the least explored kpoint
    kn1 = generate_another_zero(k,hessfun,dk=rdk(),**kwargs) # new guess
    kn2 = generate_another_zero(k,hessfun,dk=-rdk(),**kwargs) # new guess
    kns = np.array([kn1,kn2]) # new points to try
    if ks2 is None: ks12 = ks1 # only the zeros
    else: ks12 = np.concatenate([ks1,ks2]) # both zeros and explored ones
    kn = select_unexplored_kpoint(ks12,kns) # least explored one from suggestion
    if ks2 is None: ks2 = np.array([kn]) # overwrite
    else: ks2 = np.concatenate([ks2,[kn]]) # store explored ones
    return kn,ks2 # return result


def generate_another_zero(k,hess,dk=0.1):
    """Given a kpoint where the function is zero and a Hessian,
    generate another kpoint where the function is zero
    following the zero direction of the hessian"""
    m = hess(k) # generate hessian matrix
    (eigs,vs) = lg.eigh(m) # eigenvalues and eigenvectors
#    print("Hess eig",eigs)
    ind = np.argmin(np.abs(eigs)) # minimum eigenvalue index
    v = vs[:,ind] # direction of the zero direction
    return k + dk*v # return new suggested kpoint







@jit(nopython=True)
def select_unexplored_kpoint(ks1,ks2):
    """Given two lists of kpoints, select the kpoint
    from the second list which is the furthest from
    the first list"""
    nk1 = len(ks1) # number of kpoints
    nk2 = len(ks2) # number of kpoints
    dmax = 100 # maximum distance
    zs2 = np.exp(1j*ks2) # in complex plane
    zs1 = np.exp(1j*ks1) # in complex plane
    dsmin = np.zeros(nk2,dtype=np.float64) # minimum distances
    jsmin = np.zeros(nk2,dtype=np.int64) # minimum indexes
    for i in range(nk2): # loop over second kpoints
        zi = zs2[i] # in complex plane
        dzmin = 10 # minimum distance
        for j in range(nk1): # loop over the first list
            zj = zs1[j] # in complex plane
            dzj = np.mean(np.abs(zi-zj)) # distance in complex plane
            if dzj<dzmin: # minimum distance 
                jmin = j # store 
                dzmin = dzj # store
        jsmin[i] = jmin # store index
        dsmin[i] = dzmin # store distance
    ialone = np.argmax(dsmin) # index of the most isolated kpoint
    return ks2[ialone] # return the most unexplored kpoint



