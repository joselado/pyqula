# routines to align spins in the right direction
import numpy as np

tol = 1e-5 # tolerancy

def most_perpendicular_vector(vs):
    """Compute what is the most perpendicular vector to a set of vectors
    vs"""
    from scipy.optimize import minimize
    def fun(thetaphi):
        return perp_jax(thetaphi,vs)
    def grad_fun(thetaphi):
        return jac_jax(thetaphi,vs)
    x0 = np.random.random(2)
    res = minimize(fun,x0,tol=tol,jac=grad_fun)
    v = tp2v(res.x) # get the vector
    return v # return the vector


def most_perp_basis(vs):
    """Rotate the magnetization to the basis in which the magnetization
    is mostly contained in the xy plane"""
    v = most_perpendicular_vector(vs)
    out = random_perp_basis(v) # create change of basis
    ovs = np.array([out.T@v for v in vs])
#    print(v)
#    print(np.round(out,2))
#    print(ovs.shape)
    return np.array(ovs)




def random_perp_basis(v):
    """Generate a random perpendicular vector"""
    out = [v] # initial vector
    T = []
    for i in range(2):
        v0 = np.random.random(3) - 0.5 # random vector
        for vi in out:
            v0 = v0/np.sqrt(v0.dot(v0)) # normalize
            v0 = v0 - np.dot(v0,vi)*vi # remove component
        v0 = v0/np.sqrt(v0.dot(v0)) # normalize
#        print(i,v0)
        out.append(v0) # store
        T.append(v0)
    T.append(v)
    return np.array(T).T # new basis



def tp2v(thetaphi):
    """Convert Theta Phi to a vector"""
    theta = thetaphi[0]
    phi = thetaphi[1]
    vx = np.sin(theta)*np.cos(phi) # Mx
    vy = np.sin(theta)*np.sin(phi) # My
    vz = np.cos(theta) # Mz
    v0 = np.array([vx,vy,vz]) # full vector
    return v0
  
import jax.numpy as jnp
from jax import jit
from jax import grad

def perp_jax_master(thetaphi,vs):
    theta = thetaphi[0]
    phi = thetaphi[1]
    vx = jnp.sin(theta)*jnp.cos(phi) # Mx
    vy = jnp.sin(theta)*jnp.sin(phi) # My
    vz = jnp.cos(theta) # Mz
    v0 = jnp.array([vx,vy,vz]) # full vector
    n = len(vs) # loop over vectors
    out = 0. # storage
    for i in range(n): # loop over vectors
        v = vs[i,:] # get this vector
        w = jnp.cross(v,v0) # cross product
        out = out - jnp.sum(w*w) # compute the norm
    return out

perp_jax = jit(perp_jax_master) # jit jax function for energy
jac_jax = jit(grad(perp_jax_master,argnums=0)) # jit jax gradient
