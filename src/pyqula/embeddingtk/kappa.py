# summy class for the kappa method

import numpy as np

class DualLocalProbe():
    def __init__(self,LP):
        self.SC_LP = LP.copy() 
        self.normal_LP = LP.copy()
        self.normal_LP.remove_pairing()
    def set_i(self,i):
        self.SC_LP.i = i
        self.normal_LP.i = i
    def get_kappa(self,**kwargs):
        from ..transporttk.kappa import get_kappa_ratio
        return get_kappa_ratio(self,**kwargs)




def get_kappa(self,T=1e-2,write=True,nsuper=1,**kwargs):
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
    dlp = DualLocalProbe(lp) # create a dual local probe
    # now that the methods are overwritten, lets compute
    ks = [] # conductances
    for i in range(len(g.r)): # loop over positions
        dlp.set_i(i) # update position
        ks.append(dlp.get_kappa(**kwargs)) # compute this site
    if write:
        np.savetxt("KAPPAS.OUT",np.array([g.r[:,0],g.r[:,1],np.array(ks)]).T)
    return g.r[:,0],g.r[:,1],np.array(ks)







