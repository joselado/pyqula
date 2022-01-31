import numpy as np
from numba import jit
from ..spectrum import eigenvalues_kmesh
from .. import parallel

# routines to compute the dielectric response function solely from eigenvalues
# note that this will ignore form factors coming from
# wavefunctions!


def epsilonk(h,energies=np.linspace(-4.0,4.0,100),nk=40,delta=2e-3,
        qs=None):
    """Compute the dielectric response function in reciprocal space"""
    es = eigenvalues_kmesh(h,nk=nk) # get eigenvalues in a mesh
    fp = lambda omega: epsilonk_fs_2d(es,omega,delta,qs=qs,nk=nk)
    out = parallel.pcall(fp,energies) # call in parallel
    return np.array(out)



def epsilonk_fs_2d(es,omega,delta,qs=None,nk=40):
    """Compute the response function in a 2d grid"""
    print("Doing",omega)
    out = epsilonk_2d_jit(es,omega,delta,np.zeros((nk,nk)))
    fs = fermisurface_2d_jit(es,omega,delta,np.zeros((nk,nk)))
    if qs is not None: 
        from ..interpolation import periodic_grid2mesh
        out = periodic_grid2mesh(out,qs)
        fs = periodic_grid2mesh(fs,qs)
    return out,fs



@jit(nopython=True)
def epsilonk_2d_jit(es,omega,delta,out):
    """Compute the response function in a 2d grid"""
    nx = es.shape[0] # nx
    ny = es.shape[1] # ny
    ne = es.shape[2] # number of energies
    for i in range(nx): # loop over kx
      for j in range(ny): # loop over ky
        for ii in range(nx): # loop over kx
          for jj in range(ny): # loop over ky
              o = 0.0 # initialize
              ei = es[(i+ii)%nx,(j+jj)%nx,:] # list of energies
              ej = es[i,j,:] # list of energies
              for ie in range(ne): # loop over energies
                for je in range(ne): # loop over energies
                    if ei[ie]<omega and ej[je]>omega:
                      o = delta/((ei[ie]-ej[je])**2+delta*delta) + o
              out[ii,jj] += o # add contribution
    return out # return output



@jit(nopython=True)
def fermisurface_2d_jit(es,omega,delta,out):
    """Compute the response function in a 2d grid"""
    nx = es.shape[0] # nx
    ny = es.shape[1] # ny
    ne = es.shape[2] # number of energies
    for i in range(nx): # loop over kx
      for j in range(ny): # loop over ky
         o = 0.0 # initialize
         ei = es[i,j,:] # list of energies
         for ie in range(ne): # loop over energies
            o = delta/((ei[ie]-omega)**2+delta*delta) + o
         out[i,j] = o # add contribution
    return out # return output


