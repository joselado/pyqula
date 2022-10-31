from . import fullgreen
import numpy as np
from ..increase_hilbert import full2profile as spatial_dos

def ldos(ht,operator=None,write=True,nsuper=None,kpath=None,**kwargs):
    """Compute the local density of states"""
    def get(ht):
        if ht.dimensionality!=1: raise
        if ht.block_diagonal:
            nc = len(ht.central_intra) # number of cells in the middle
            ls = [] # storage for LDOS
            if operator is not None:
                op = ht.Hr.get_operator(operator) # redefine operator
            for ic in range(nc): # loop
                g = fullgreen.get_full_green(ht,mode="central",ic=ic,**kwargs)
                if operator is not None: g = op*g # redefine
                d = [ -(g[i,i]).imag/np.pi for i in range(len(g))] # LDOS
                d = spatial_dos(ht.Hr,d) # convert to spatial resolved DOS
                ls = np.concatenate([ls,d]) # store LDOS
        else: raise
        return nc,ls # return positions and LDOS
    if ht.dimensionality==1: 
        nc,ls = get(ht) # 1d
        r = ht.Hr.geometry.get_supercell(nc).r # generate supercell
    elif ht.dimensionality==2: 
        if kpath is None: kpath = np.linspace(0.,1.,200)
        from ..parallel import pcall
        out = pcall(lambda k: get(ht.generate(k)),kpath) # parallelize over k
        nc = out[0][0] # positions
        ls = 0.
        for i in range(len(out)): ls = ls + out[i][1]
        ls = ls/len(kpath) # normalize
        # now generate the positions
        g = ht.Hr.geometry # original geometry
        from .. import sculpt
        g = sculpt.rotate_a2b(g,g.a2,np.array([1.0,0.0,0.0]))
        gs = g.get_supercell([nc,1]) # supercell in junction dir
        if nsuper is None: nsuper = nc
        r = gs.get_supercell([1,nsuper]).r # supercell in invariant dir
        # now replicate the LDOS
        lso = []
        for i in range(nsuper): lso = np.concatenate([lso,ls])
        ls = lso # overwrite
    else: raise
    if write:
        np.savetxt("LDOS.OUT",np.array([r[:,0],r[:,1],ls]).T)
    return r[:,0],r[:,1],ls





