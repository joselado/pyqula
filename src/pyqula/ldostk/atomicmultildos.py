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
        ratomic=1.5,dr=0.2,num_bands=None,**kwargs):
    """Compute the LDOS at different eenrgies, and add an envelop atomic
    orbital"""
    h = h.copy() # copy the Hamiltonian
    h.turn_dense() # dense hamiltonian
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
    x,y = get_grids(h.geometry,nrep=nrep,dr=dr,
            deltax=ratomic,deltay=ratomic)
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
    ds = np.zeros((len(vs),len(x))) # zero array
    # get the generator
    density_generator = get_real_space_density_generator(lodict,h.geometry,
                              has_spin=h.has_spin)
    for i in range(len(vs)): # loop over wavefunctions
        w = vs[i] # get the current Bloch wavefunction
        k = ks[i] # get the current bloch wavevector
        d = density_generator(w,k)
        ds[i] = d # store in the list
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




def multi_ldos(h,es=np.linspace(-2.0,2.0,100),delta=0.05,**kwargs):
    """Compute the LDOS at different eenrgies, and add an envelop atomic
    orbital"""
    ldos_gen,evals,x,y = ldos_generator(h,delta=delta,**kwargs) # get the generator
    # now compute all the LDOS
    fs.rmdir("MULTILDOS")
    fs.mkdir("MULTILDOS")
    fo = open("MULTILDOS/MULTILDOS.TXT","w") # files with the names
    for e in es: # loop over energies
        name0 = "LDOS_"+str(e)+"_.OUT" # name of the output
        name = "MULTILDOS/" + name0
        out = ldos_gen(e) # compute the LDOS
        np.savetxt(name,np.array([x,y,out]).T) # save
        fo.write(name0+"\n") # name of the file
    fo.close()
    from ..dos import calculate_dos,write_dos
    es2 = np.linspace(min(es),max(es),len(es)*10)
    ys = calculate_dos(evals,es2,delta,w=None) # compute DOS
    write_dos(es2,ys,output_file="MULTILDOS/DOS.OUT")



def get_real_space_density_generator(lodict,g,has_spin=False):
    """Compute the orbital in real space"""
    out = 0. # wavefunction in real space
    orbs = np.array([lodict[key] for key in lodict]) # orbitals
    inds = np.array([key[1] for key in lodict],dtype=int) # indexes
    ds = [key[0] for key in lodict]
    def fout(w,k):
        """Function that return the real space density"""
        nc = len(w) # number of components of the Bloch wavefunction
        phis = np.array([g.bloch_phase(d,k) for d in ds]) # phases
        out = np.zeros(orbs[0].shape[0],dtype=np.complex)
        if not has_spin: # spinless
          return get_real_space_density_jit(w,phis,inds,orbs,out).real
        else: # spinful
          wup = ud_component(w,mode="up") 
          wdn = ud_component(w,mode="dn") 
          outup = get_real_space_density_jit(wup,phis,inds,orbs,out).real
          outdn = get_real_space_density_jit(wdn,phis,inds,orbs,out).real
          return outup + outdn
    return fout # return function

@jit(nopython=True)
def get_real_space_density_jit(w,phis,inds,orbs,out):
    """Return the real space wavefunction"""
    for ii in range(len(inds)):
        iorb = int(inds[ii]) # integer
        out = out + w[iorb]*phis[ii]*orbs[ii] # add contribution
    return out*np.conjugate(out)




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



def ud_component(w,mode="up"):
    """Given a wavefunction, return the up components"""
    n = len(w)//2
    wo = w.copy()+0j # copy
    if mode=="up": p = 0
    elif mode=="dn": p = 1
    else: raise
    for i in range(n): wo[2*i+p] = 0.0j # set to zero
    return wo

