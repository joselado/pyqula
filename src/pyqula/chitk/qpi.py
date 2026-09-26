# library to deal with the spectral properties of the hamiltonian
import numpy as np
import scipy.linalg as lg
import scipy.sparse.linalg as slg
import os
from numba import jit
from .. import filesystem as fs
from .. import parallel
from .. import interpolation


def get_qpi(h,reciprocal=True,nk=20,energies=np.linspace(-4.0,4.0,80),
        output_folder="MULTIQPI",nsuper=2,integrate=False,
        mode = "response",info=False,
        nunfold = 1, # flag for unfolding
        delta=1e-1,**kwargs):
    """Compute the QPI using a poor-mans convolution of the k-DOS"""
    if h.dimensionality!=2:
        raise ValueError("the QPI is only implemented for 2d Hamiltonians")
    if mode=="response": # built from the bare eigenvalues alone
        # it takes no operator and does no unfolding, and both used to be
        # dropped without a word, returning the plain QPI of the supercell
        if kwargs.get("operator") is not None or nunfold!=1:
            raise NotImplementedError("get_qpi(mode='response') is built "
                    "from the eigenvalues alone and takes neither an "
                    "operator nor nunfold (got operator="
                    +str(kwargs.get("operator"))+", nunfold="+str(nunfold)
                    +"); use mode='pm' to weight or unfold the QPI")
    unfold = mode=="pm" and nunfold!=1 # unfold onto the primal zone
    # the convolution below reads the q-points in the coordinates of the
    # k-mesh, which are the primal cell's when unfolding, so they are
    # mapped with the primal geometry then (the same map for an n x n
    # supercell, a rotated one for the sqrt(3) x sqrt(3) cell)
    gq = check_nunfold(h,nunfold) if unfold else h.geometry
    if reciprocal: fR = gq.get_k2K_generator() # get matrix
    else:  fR = lambda x: x # get identity
    qs0 = h.geometry.get_kmesh(nk=nk*nsuper,nsuper=nsuper)
    qs0 = qs0 - np.mean(qs0,axis=0)
    qs = np.array([fR(q) for q in qs0]) # convert
    if mode=="pm": # poor man mode
        from ..fermisurface import fermi_surface_generator
        # when unfolding, the mesh is [0,1]^2 in the reduced coordinates
        # of the primal cell, computed at its image M@k in the supercell,
        # so ks are primal coordinates whatever the supercell matrix M.
        # This used to sample [0,nunfold]^2 in the supercell's coordinates
        # and divide by nunfold, the same mesh for an nunfold x nunfold
        # supercell and the wrong one for any other
        es,ks,ds = fermi_surface_generator(h,reciprocal=False,info=info,
                energies=energies,delta=delta,
                full_bz=True, # the whole first Brillouin zone
                primal_mesh=unfold,nsuper=1,nk=nk,**kwargs)
        # we now have the energies, k-points and DOS, lets do a convolution
        fp = lambda i: poor_man_qpi_single_energy(ks,ds[:,i],qs) # parallel function
        out = parallel.pcall(fp,range(len(es))) # compute in parallel
        dosa = np.sum(ds,axis=0) # array for the DOS
    ### alternative method ###
    elif mode=="response":
        from .epsilon import epsilonk
        out = epsilonk(h,energies=energies,nk=nk,delta=delta,qs=qs) # output
        es = energies # redefine the energies
        dosa = np.sum([o[1] for o in out],axis=1) # DOS
    else:
        raise ValueError("unknown mode; the QPI accepts 'pm' and 'response'")
#    print(np.array(out).shape) ; exit()
    # now write everything #
    ########################################
    fs.rmdir(output_folder) # remove folder
    fs.mkdir(output_folder) # create folder
    kqpi = np.array([o[0] for o in out]).T # convert to array
    if integrate:
        kqpi = [np.mean(kqpi[:,0:i],axis=1) for i in range(len(es))]
        kqpi = [kp-np.min(kp) for kp in kqpi]
        kqpi = np.array(kqpi).T
    kdos = np.array([o[1] for o in out]).T # convert to array
    fo = open(output_folder+"/"+output_folder+".TXT","w")
    for i in range(len(es)): # loop over energies
        filename = output_folder+"_"+str(es[i])+"_.OUT" # name
        name = output_folder+"/"+filename
        np.savetxt(name,np.array([qs0[:,0],qs0[:,1],kqpi[:,i]]).T)
        np.savetxt(name+"_FS",np.array([qs0[:,0],qs0[:,1],kdos[:,i]]).T)
        fo.write(filename+"\n")
        name = output_folder+"/DOS.OUT"
    name = "DOS.OUT"
    np.savetxt(name,np.array([es,dosa]).T)
    fo.close()



def check_nunfold(h,nunfold):
    """Return the primal geometry the QPI is unfolded onto, after checking
    that nunfold is the size of the supercell. The supercell matrix itself
    is read from the geometry, so nunfold only has to agree with it, and it
    is read as get_supercell reads a size: the linear size, nunfold**2
    primal cells, so np.sqrt(3) for the sqrt(3) x sqrt(3) cell and n for
    an n x n one"""
    g0 = getattr(h.geometry,"primal_geometry",None)
    if g0 is None:
        raise ValueError("get_qpi unfolds (nunfold="+str(nunfold)+") only "
                "a supercell built with store_primal=True, which the "
                "unfolding operator needs as well")
    from ..unfolding import get_supercell_map
    M = get_supercell_map(h.geometry,g0)[0] # A_S = M@A_0
    ncells = abs(np.linalg.det(np.array(M,dtype=float)))
    if abs(nunfold**2-ncells)>1e-6*ncells:
        raise ValueError("nunfold="+str(nunfold)+" does not match this "
            "supercell, which holds "+str(int(round(ncells)))+" primal "
            "cells; nunfold is its linear size, the square root of that, "
            "as in get_supercell (np.sqrt(3) for the sqrt(3) x sqrt(3) "
            "cell)")
    return g0



def poor_man_qpi_single_energy(ks,ds,qs):
#    return poor_man_qpi_single_energy_brute_force(ks,ds,qs)
    return poor_man_qpi_convolve(ks,ds,qs)



def poor_man_qpi_convolve(ks,ds,qs):
    """Convolve the DOS to simmulate the QPI"""
    nq = int(np.sqrt(len(qs))) # number of qpoints
    nk = int(np.sqrt(len(ks))) # number of qpoints
    grid_kx, grid_ky = np.mgrid[0:1:nq*1j, 0:1:nq*1j] # kx and ky
    # DOS in a grid
    f = interpolation.interpolator2d(ks[:,0],ks[:,1],ds,mode="periodic")
    ksg = np.array([grid_kx, grid_ky]).reshape((2,nq*nq)).T # create points
    dsg = f(ksg).reshape((nq,nq)) # interpolate on a grid
    from ..convolution import selfconvolve
    out = selfconvolve(dsg).reshape((nq*nq)) # do the convolution
    out = interpolation.interpolator2d(ksg[:,0],ksg[:,1],out,mode="periodic")(qs[:,0:2])
    dsout = interpolation.interpolator2d(ksg[:,0],ksg[:,1],f(ksg),mode="periodic")(qs[:,0:2])
    print("Done QPI")
    return out,dsout


def poor_man_qpi_single_energy_brute_force(ks,ds,qs):
    """Do the convolution of the Fermi surfaces"""
    f0 = interpolation.interpolator2d(ks[:,0],ks[:,1],ds) # interpolated k-DOS
    def f(k):
        """Define a periodic function"""
        k = k[:,0:2]%1.
        o = f0(k)
        return o
    print("Doing")
    out = [np.mean(f(ks)*f(ks+q)) for q in qs]
    return np.array(out) # return output







