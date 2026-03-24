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
#    gradfun = None
    from scipy.optimize import minimize
    tol = 1e-4
    outs = [] # empty list
    k0 = np.random.random(H.dimensionality) - 0.5 # starting point
    for i in range(nk):
#        print(k0) ; exit()
        result = minimize(fun,k0,tol=tol,jac=gradfun,hess=hessfun,
                          method="SLSQP",
                          bounds=[[-1.,1.],[-1.,1.]])
        if i%20==0:
            k0 = np.random.random(H.dimensionality) - 0.5 # starting point
        else: k0 = result.x + 0.1*(np.random.random(H.dimensionality) - 0.5)
        if np.exp(fun(result.x))<tol:
            o = result.x%1. # output
            for j1 in range(nrep): # loop
                for j2 in range(nrep): # loop
                    op = o + np.array([j1,j2])
                    outs.append(R@op) # store
    return np.array(outs) # return kpoints


