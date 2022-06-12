import numpy as np

# routines to compute the dIdV using the embedding method

def get_didv(self,T=1e-2,write=True,nsuper=1,**kwargs):
    from ..transporttk.localprobe import LocalProbe
    lp = LocalProbe(self,T=T,**kwargs) # local probe object
    lp.reuse_gf = True # reuse the Green's function
    # now we will overwrite a few objects
    # this is not very elegant, but it works
    g = lp.H.geometry.get_supercell(nsuper) # supercell geometry
    Hc = lp.H.copy() # copy the original object
    # for the selfenernergy, the intracell is picked from lp.H
    lp.H = self.H.get_supercell(nsuper) # overwrite Hamiltonian (for the intra)
    # the Green's function is now directly computed for the supercell
    lp.H.get_gf = lambda **kwargs: Hc.get_gf(nsuper=nsuper,**kwargs)
    # now that the methods are overwritten, lets compute
    Gs = [] # conductances
    for i in range(len(g.r)): # loop over positions
        lp.i = i # update position
        Gs.append(lp.didv(**kwargs)) # compute this site
    if write:
        np.savetxt("DIDV.OUT",np.array([g.r[:,0],g.r[:,1],np.array(Gs)]).T)
    return g.r[:,0],g.r[:,1],np.array(Gs)


