import numpy as np
from numba import jit
import os
from .. import filesystem as fs
from numba import jit


@jit(nopython=True)
def jit_get_orbital(r0,rs,ratomic): 
    """Atomic orbital"""
    dr = rs-r0
    dr2 = np.sum(dr*dr,axis=1) # sum
    return np.exp(-np.sqrt(dr2)/ratomic)


def ldos_generator(h,**kwargs):
    return profile_generator(h,mode="LDOS",**kwargs)


def profile_generator(h,delta=0.05,nrep=1,nk=20,dl=None,mode="LDOS",
        ratomic=1.5,dr=0.2,num_bands=None,
        deltax=None, deltay=None,**kwargs):
    """Compute the LDOS at different eenrgies, and add an envelop atomic
    orbital"""
    h = h.copy() # copy the Hamiltonian
    h = h.get_dense() # dense hamiltonian
    evals,vs,ks = h.get_eigenvectors(nk=nk,kpoints=True,
            numw=num_bands) # compute wavefunctions
    if dl is None: 
        if h.dimensionality==0: nrepdl = int(ratomic*10)
        dl = h.geometry.neighbor_directions(nrep+int(ratomic*10)) # directions of the neighbors
    def get_orbital(r0,r):
        return jit_get_orbital(r0,r,ratomic)
    # generate a dictionary with all the real space local orbitals
    ##########################################################
    lodict = dict() # dictionary for the local orbitals
    # get the grids
    if deltax is None: deltax=ratomic
    if deltay is None: deltay=ratomic
    x,y = get_grids(h.geometry,nrep=nrep,dr=dr,
            deltax=deltax,deltay=deltay)
    r = np.zeros((len(x),3)) ; r[:,0] = x ; r[:,1] = y
    # now chack which centers to accept
    xmin,xmax = np.min(x),np.max(x)
    ymin,ymax = np.min(y),np.max(y)
    def accept_center(r):
        """Check if this center is close enough"""
        fac = 10
        if r[0]-xmin<-(fac*ratomic): return False # too left
        if r[0]-xmax>(fac*ratomic): return False # too right
        if r[1]-ymin<-(fac*ratomic): return False # too down
        if r[1]-ymax>(fac*ratomic): return False # too right
        return True
    for d in dl: # loop over directions
          rrep = h.geometry.replicas(d=d) # replicas in this direction
          for i in range(len(rrep)): # loop over the atoms
              r0 = rrep[i] # get this center
              if not accept_center(r0): continue # skip this iteration
              if h.has_eh:
                if h.has_spin: # spinful
                  lodict[(tuple(d),4*i)] = get_orbital(r0,r) # store 
                  lodict[(tuple(d),4*i+1)] = get_orbital(r0,r) # store 
           #       lodict[(tuple(d),4*i+2)] = 0. # store 
           #       lodict[(tuple(d),4*i+3)] = 0. # store 
                else: raise
              else:
                if h.has_spin: # spinful
                  lodict[(tuple(d),2*i)] = get_orbital(r0,r) # store 
                  lodict[(tuple(d),2*i+1)] = get_orbital(r0,r) # store 
                else: # spinless
                  lodict[(tuple(d),i)] = get_orbital(r0,r) # store 
    ##########################################################
    # now compute the real-space wavefunctions including the Bloch phase
    ds = get_real_space_density_batch(lodict,h.geometry,vs,ks,
                              has_spin=h.has_spin) # (nwf,ngrid) array
    if mode=="LDOS": # LDOS mode
      def f(e): return ldos_at_energy(evals,ds,e,delta) # compute the LDOS
    elif mode=="density": # LDOS mode
      def f(e): return density_at_energy(evals,ds,e,delta) # compute the LDOS
    else: raise # not implemented
    return f,evals,x,y # return generator


def get_ldos(h,e=0.0,delta=0.05,**kwargs):
    """Compute a single LDOS"""
    ldos_gen,evals,x,y = ldos_generator(h,e=e,delta=delta,**kwargs) 
    out = ldos_gen(e) # compute the LDOS
    np.savetxt("LDOS.OUT",np.array([x,y,out]).T) # save
    return x,y,out


def get_density(h,e=0.0,delta=1e-3,**kwargs):
    """Compute a single LDOS"""
    ldos_gen,evals,x,y = profile_generator(h,e=e,delta=delta,
                               mode="density",**kwargs)
    out = ldos_gen(e) # compute the LDOS
    np.savetxt("DENSITY.OUT",np.array([x,y,out]).T) # save
    return x,y,out




def multi_ldos(h,energies=np.linspace(-2.0,2.0,100),delta=0.05,**kwargs):
    """Compute the LDOS at different eenrgies, and add an envelop atomic
    orbital"""
    ldos_gen,evals,x,y = ldos_generator(h,delta=delta,**kwargs) # get the generator
    # now compute all the LDOS
    fs.rmdir("MULTILDOS")
    fs.mkdir("MULTILDOS")
    fo = open("MULTILDOS/MULTILDOS.TXT","w") # files with the names
    for e in energies: # loop over energies
        name0 = "LDOS_"+str(e)+"_.OUT" # name of the output
        name = "MULTILDOS/" + name0
        out = ldos_gen(e) # compute the LDOS
        np.savetxt(name,np.array([x,y,out]).T) # save
        fo.write(name0+"\n") # name of the file
    fo.close()
    from ..dos import calculate_dos,write_dos
    es2 = np.linspace(min(energies),max(energies),len(energies)*10)
    ys = calculate_dos(evals,es2,delta,w=None) # compute DOS
    write_dos(es2,ys,output_file="MULTILDOS/DOS.OUT")



def get_real_space_density_batch(lodict,g,vs,ks,has_spin=False):
    """Compute the real-space density for all the wavefunctions at once.

    Every eigenvector is expanded in the same fixed set of atomic orbitals
    (one row of ``orbs`` per (direction,orbital) key), so the whole batch of
    wavefunctions can be projected with a single dense matrix product
    (BLAS ``zgemm``) instead of one Python/numba call per wavefunction. The
    Bloch phase for a given k-point is also cached, since many wavefunctions
    (all the bands at the same k) share it."""
    orbs = np.array([lodict[key] for key in lodict]) # (nentries,ngrid)
    inds = np.array([key[1] for key in lodict],dtype=int) # (nentries,)
    dirs = [key[0] for key in lodict]
    vs = np.array(vs,dtype=np.complex128) # (nwf,norb)
    phis_cache = dict() # cache the Bloch phases, shared across bands at a k
    phis = np.empty((len(ks),len(dirs)),dtype=np.complex128)
    for i,k in enumerate(ks):
        key = tuple(k)
        cached = phis_cache.get(key)
        if cached is None:
            cached = np.array([g.bloch_phase(d,k) for d in dirs])
            phis_cache[key] = cached
        phis[i] = cached
    def project(w):
        """Coefficients of each wavefunction in the fixed orbital basis"""
        c = w[:,inds]*phis # (nwf,nentries)
        psi = c@orbs # (nwf,ngrid) single dense matrix product
        return (psi*np.conjugate(psi)).real
    if not has_spin: # spinless
        return project(vs)
    else: # spinful: sum the up and down spin-channel densities
        wup = vs.copy() ; wup[:,0::2] = 0.0j
        wdn = vs.copy() ; wdn[:,1::2] = 0.0j
        return project(wup) + project(wdn)




def get_grids(g,nrep=1,dr=0.1,deltax=1.0,deltay=1.0):
    """Return the grids to plot the real space wavefunctions"""
    r = g.multireplicas(nrep-1) # get all the position
    xmin = np.min(r[:,0])
    xmax = np.max(r[:,0])
    ymin = np.min(r[:,1])
    ymax = np.max(r[:,1])
    nx = int((xmax-xmin+2*deltax)/dr) # number of x points
    ny = int((ymax-ymin+2*deltay)/dr) # number of y points
    xp = np.linspace(xmin-deltax,xmax+deltax,nx) # generate the points
    yp = np.linspace(ymin-deltay,ymax+deltay,ny) # generate the points
    gridx = np.zeros(nx*ny)
    gridy = np.zeros(nx*ny)
    gridx,gridy = get_grids_jit(xp,yp,gridx,gridy)
    return gridx,gridy # return the grids

@jit(nopython=True)
def get_grids_jit(x,y,gridx,gridy):
    nx = len(x)
    ny = len(y)
    k = 0
    for i in range(nx):
      for j in range(ny):
          gridx[k] = x[i]
          gridy[k] = y[j]
          k += 1
    return gridx,gridy

def ldos_at_energy(evals,ds,e,delta):
    """Compute the different local density of states at each energy"""
    de2 = (evals-e)**2 # difference in energy
    out = np.sum(ds.T*delta/(de2+delta**2),axis=1)
    return out # return that density

def density_at_energy(evals,ds,e,delta):
    """Compute the density at this energy"""
    de = evals-e # difference in energy
    w = (1. - np.tanh(de/delta))/2. # weight
    out = np.sum(ds.T*w,axis=1) # output
    return out # return that density


