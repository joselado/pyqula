import numpy as np

# routines to compute the dIdV using the embedding method

def get_didv(self,T=1e-2,write=True,nsuper=1,**kwargs):
    from ..transporttk.localprobe import LocalProbe
    # Build LocalProbe from the pristine Hamiltonian (self.H), not the
    # Embedding object itself: LocalProbe.__init__ needs real Hamiltonian
    # attributes (is_multicell, get_no_multicell, intra/tx/ty for
    # make_compatible) that Embedding doesn't provide, and would crash
    # with AttributeError otherwise. lp.H gets overwritten below anyway,
    # so this changes nothing about what LocalProbe ends up using.
    Hc = self.copy() # copy of the embedding object itself (defect + selfenergy)
    lp = LocalProbe(self.H,T=T,**kwargs) # local probe object
    lp.reuse_gf = True # reuse the Green's function
    # the probe-lead selfenergy (lead=0) doesn't depend on the site index
    # i (see LocalProbe.get_selfenergy), so it only needs to be solved
    # once here and reused for every site below, instead of redone via a
    # full Sancho-Rubio renormalization on each one of potentially many
    # sites in the map.
    lp.reuse_selfenergy = True
    # now we will overwrite a few objects
    # this is not very elegant, but it works
    g = self.H.geometry.get_supercell(nsuper) # supercell geometry
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




def get_didv_single(self,T=1e-2,write=True,i=0,nsuper=1,**kwargs):
    from ..transporttk.localprobe import LocalProbe
    # see get_didv above for why LocalProbe is built from self.H (the
    # pristine Hamiltonian) rather than the Embedding object itself
    Hc = self.copy() # copy of the embedding object itself (defect + selfenergy)
    lp = LocalProbe(self.H,T=T,**kwargs) # local probe object
    lp.reuse_gf = True # reuse the Green's function
    # now we will overwrite a few objects
    # this is not very elegant, but it works
    g = self.H.geometry.get_supercell(nsuper) # supercell geometry
    # for the selfenernergy, the intracell is picked from lp.H
    lp.H = self.H.get_supercell(nsuper) # overwrite Hamiltonian (for the intra)
    # the Green's function is now directly computed for the supercell
    lp.H.get_gf = lambda **kwargs: Hc.get_gf(nsuper=nsuper,**kwargs)
    # now that the methods are overwritten, lets compute
    Gs = [] # conductances
    lp.i = i # update position
    return lp.didv(**kwargs) # compute this site

