import numpy as np
from numba import jit

import scipy.linalg as lg

#use_fortran = False
try:
    from . import dyson2df90
    use_fortran = True
except:
    use_fortran = False

use_fortran = False # do not use the Fortran routines

def dyson(h0,nsuper,nk,ez):
    try: # try to use the non multicell method
        h = h0.get_no_multicell()
    except: # if not possible, try the multicell one (not well tested yet)
        h = h0.copy()
    h = h.get_dense() # dense matrices
    if h.dimensionality==0: raise
    if not h.is_multicell: # no multicell
        if h.dimensionality==1: 
            return dyson1d(h.intra,h.inter,nsuper[0],nk,ez)
        elif h.dimensionality==2: 
            return dyson2d(h.intra,h.tx,h.ty,h.txy,h.txmy,nsuper[0],nsuper[1],nk,ez)
        else: raise
    else:
        print("WARNING, Multicell function in Dyson")
        hkgen = h.get_hk_gen() # get Hamiltonian generator
        if h.dimensionality==1: 
            return dyson1d_hkgen(hkgen,nsuper[0],nk,ez)
        elif h.dimensionality==2: 
            return dyson2d_hkgen(hkgen,nsuper[0],nsuper[1],nk,ez)
        else: raise




def dyson2d(intra,tx,ty,txy,txmy,nx,ny,nk,ez):
    if use_fortran:
        from . import dyson2df90
        return dyson2df90.dyson2d(intra,tx,ty,txy,txmy,nx,ny,nk,ez)
    else:
        ns = intra.shape[0]*nx*ny
        g = np.zeros((ns,ns),dtype=np.complex128)
        nkx,nky = nk,nk
        return dyson2d_jit(intra,tx,ty,txy,txmy,nx,ny,nkx,nky,ez,g)


def dyson2d_hkgen(hkgen,nx,ny,nk,ez):
    ns = hkgen(np.array([0.,0.])).shape[0]*nx*ny
    g = np.zeros((ns,ns),dtype=np.complex128)
    nkx,nky = nk,nk
#    return dyson2d_hkgen_jit(hkgen,nx,ny,nkx,nky,ez,g)
    gsk = generate_gfk(hkgen,nx,ny,nkx,nky,ez) # generate Green's functions
    return dyson2d_gsk_jit(gsk,nx,ny,nkx,nky,ez,g) # generate the full Gf from the individual ones




def dyson1d(intra,inter,nx,nkx,ez):
    """Workaround for 1D"""
    zero = intra*0.0 # zero matrix
    ns = intra.shape[0]*nx
    g = np.zeros((ns,ns),dtype=np.complex128)
    return dyson2d_jit(intra,inter,zero,zero,zero,nx,1,nkx,1,ez,g)


def dyson1d_hkgen(hkgen,nx,nkx,ez):
    """Workaround for 1D"""
    zero = hkgen([0.])*0.0 # zero matrix
    ns = zero.shape[0]*nx
    g = np.zeros((ns,ns),dtype=np.complex128)
    return dyson2d_jit(hkgen,nx,1,nkx,1,ez,g)






@jit(nopython=True)
def dyson2d_jit(intra,tx,ty,txy,txmy,nx,ny,nkx,nky,ez,g):
    """jit version of the function"""
    n = intra.shape[0] # size of the matrix
    gs = np.zeros((nkx*nky,n,n),dtype=np.complex128) # GF in k-points
    gm = np.zeros((nx*2-1,ny*2-1,n,n),dtype=np.complex128) # GF in minicells
    ks = np.zeros((nkx*nky,2)) # kpoints
    em = np.identity(n) # identity times energy
    em = em*ez # identity times energy
    ik = 0
    nk2 = nkx*nky # total number of kpoints
    # Compute all the GF
    for i in range(nkx):
        kx = 1./nkx*np.pi*2*i # kpoint
        for j in range(nky):
            ky = 1./nky*np.pi*2*j # kpoint
            m = np.exp(1j*kx)*tx + np.exp(1j*ky)*ty
            m = m + np.exp(1j*(kx+ky))*txy + np.exp(1j*(kx-ky))*txmy
            m = m + np.conjugate(m.T) + intra # Bloch Hamiltonian
            gs[ik,:,:] = np.linalg.inv(em - m) # store GF
            ks[ik,0] = kx
            ks[ik,1] = ky
            ik += 1 # increase counter
    # GF in the different "minicells" of the supercell
    for i in range(-nx+1,nx):
        for j in range(-ny+1,ny):
            m = np.zeros((n,n),dtype=np.complex128) # initialize
            for ik in range(nk2): # loop over kpoints
                m = m + gs[ik,:,:]*np.exp(1j*(ks[ik,0]*i+ks[ik,1]*j))
            gm[nx+i-1,ny+j-1,:,:] = m[:,:]/nk2 # store
    # now create indexes for the minicell
    inds = np.zeros((nx*ny,2),dtype=np.int_)
    ic = 0
    for i in range(nx):
        for j in range(ny):
            inds[ic,0] = i
            inds[ic,1] = j
            ic += 1
    # now compute the supercell GF
    for i in range(nx*ny): # loop over rows
        i1 = inds[i,0] # minicell index
        j1 = inds[i,1] # minicell index
        for j in range(nx*ny):
            i2 = inds[j,0] # minicell index
            j2 = inds[j,1] # minicell index
            ii = i1-i2 + nx -1 
            jj = j1-j2 + ny -1
            m[:,:] = gm[ii,jj,:,:] # get this matrix
            ii0 = n*i
            ii1 = n*(i+1)
            jj0 = n*j
            jj1 = n*(j+1)
            g[ii0:ii1,jj0:jj1] = m[:,:] # store all this data
    return g




def generate_gfk(hkgen,nx,ny,nkx,nky,ez):
    """Generate all the required Green's functions"""
    n = hkgen([0.,0.]).shape[0] # size of the matrix
    gs = np.zeros((nkx*nky,n,n),dtype=np.complex128) # GF in k-points
    gm = np.zeros((nx*2-1,ny*2-1,n,n),dtype=np.complex128) # GF in minicells
    ks = np.zeros((nkx*nky,2)) # kpoints
    em = np.identity(n) # identity times energy
    em = em*ez # identity times energy
    ik = 0
    nk2 = nkx*nky # total number of kpoints
    # Compute all the GF
    for i in range(nkx):
        kx = 1./nkx*np.pi*2*i # kpoint
        for j in range(nky):
            ky = 1./nky*np.pi*2*j # kpoint
            k = np.array([kx,ky])/(2*np.pi) # k vector in natural
            m = hkgen(k) # compute Bloch Hamiltonian
            gs[ik,:,:] = np.linalg.inv(em - m) # store GF
            ks[ik,0] = kx
            ks[ik,1] = ky
            ik += 1 # increase counter
    return gs # return Green's functions









# this function is no longer used, but kept for testing purposes
#@jit(nopython=True)
def dyson2d_hkgen_jit(hkgen,nx,ny,nkx,nky,ez,g):
    """jit version of the function"""
    n = hkgen([0.,0.]).shape[0] # size of the matrix
    gs = np.zeros((nkx*nky,n,n),dtype=np.complex128) # GF in k-points
    gm = np.zeros((nx*2-1,ny*2-1,n,n),dtype=np.complex128) # GF in minicells
    ks = np.zeros((nkx*nky,2)) # kpoints
    em = np.identity(n) # identity times energy
    em = em*ez # identity times energy
    ik = 0
    nk2 = nkx*nky # total number of kpoints
    # Compute all the GF
    for i in range(nkx):
        kx = 1./nkx*np.pi*2*i # kpoint
        for j in range(nky):
            ky = 1./nky*np.pi*2*j # kpoint
            k = np.array([kx,ky])/(2*np.pi) # k vector in natural
            m = hkgen(k) # compute Bloch Hamiltonian
            gs[ik,:,:] = np.linalg.inv(em - m) # store GF
            ks[ik,0] = kx
            ks[ik,1] = ky
            ik += 1 # increase counter
    # GF in the different "minicells" of the supercell
    for i in range(-nx+1,nx):
        for j in range(-ny+1,ny):
            m = np.zeros((n,n),dtype=np.complex128) # initialize
            for ik in range(nk2): # loop over kpoints
                m = m + gs[ik,:,:]*np.exp(1j*(ks[ik,0]*i+ks[ik,1]*j))
            gm[nx+i-1,ny+j-1,:,:] = m[:,:]/nk2 # store
    # now create indexes for the minicell
    inds = np.zeros((nx*ny,2),dtype=np.int_)
    ic = 0
    for i in range(nx):
        for j in range(ny):
            inds[ic,0] = i
            inds[ic,1] = j
            ic += 1
    # now compute the supercell GF
    for i in range(nx*ny): # loop over rows
        i1 = inds[i,0] # minicell index
        j1 = inds[i,1] # minicell index
        for j in range(nx*ny):
            i2 = inds[j,0] # minicell index
            j2 = inds[j,1] # minicell index
            ii = i1-i2 + nx -1 
            jj = j1-j2 + ny -1
            m[:,:] = gm[ii,jj,:,:] # get this matrix
            ii0 = n*i
            ii1 = n*(i+1)
            jj0 = n*j
            jj1 = n*(j+1)
            g[ii0:ii1,jj0:jj1] = m[:,:] # store all this data
    return g




@jit(nopython=True)
def dyson2d_gsk_jit(gs,nx,ny,nkx,nky,ez,g):
    """Compute full Gf from individual ones"""
    n = gs[0].shape[0] # size of the matrix
#    gs = np.zeros((nkx*nky,n,n),dtype=np.complex128) # GF in k-points
    gm = np.zeros((nx*2-1,ny*2-1,n,n),dtype=np.complex128) # GF in minicells
    ks = np.zeros((nkx*nky,2)) # kpoints
    em = np.identity(n) # identity times energy
    em = em*ez # identity times energy
    ik = 0
    nk2 = nkx*nky # total number of kpoints
    # Compute all the GF
    for i in range(nkx):
        kx = 1./nkx*np.pi*2*i # kpoint
        for j in range(nky):
            ky = 1./nky*np.pi*2*j # kpoint
            ks[ik,0] = kx
            ks[ik,1] = ky
            ik += 1 # increase counter
    # GF in the different "minicells" of the supercell
    for i in range(-nx+1,nx):
        for j in range(-ny+1,ny):
            m = np.zeros((n,n),dtype=np.complex128) # initialize
            for ik in range(nk2): # loop over kpoints
                m = m + gs[ik,:,:]*np.exp(1j*(ks[ik,0]*i+ks[ik,1]*j))
            gm[nx+i-1,ny+j-1,:,:] = m[:,:]/nk2 # store
    # now create indexes for the minicell
    inds = np.zeros((nx*ny,2),dtype=np.int_)
    ic = 0
    for i in range(nx):
        for j in range(ny):
            inds[ic,0] = i
            inds[ic,1] = j
            ic += 1
    # now compute the supercell GF
    for i in range(nx*ny): # loop over rows
        i1 = inds[i,0] # minicell index
        j1 = inds[i,1] # minicell index
        for j in range(nx*ny):
            i2 = inds[j,0] # minicell index
            j2 = inds[j,1] # minicell index
            ii = i1-i2 + nx -1 
            jj = j1-j2 + ny -1
            m[:,:] = gm[ii,jj,:,:] # get this matrix
            ii0 = n*i
            ii1 = n*(i+1)
            jj0 = n*j
            jj1 = n*(j+1)
            g[ii0:ii1,jj0:jj1] = m[:,:] # store all this data
    return g















