import numpy as np
from . import algebra
from .chitk import pmrkky # poor man RKKY



def rkky(h,mode="pm",**kwargs):
    """Compute the RKKY interaction using different methodologies"""
    # this only works with spinless systems
    if mode=="pm": # poor man's methodology
        return pmrkky.explicit_rkky(h,**kwargs) # return explicitly the RKKY
    elif mode=="LR": # linear response theory
        from .chitk import magneticresponse
        return magneticresponse.rkky(h,**kwargs)
    else: raise



def rkky_map(h0,n=2,mode="LR",info=False,fsuper=8,**kwargs):
    """Compute the RKKY map using a brute force algorithm"""
    h = h0.copy() # copy Hamiltonian
    g0 = h.geometry.copy() # copy initial geometry
    if mode=="pm":
        h = h.get_supercell(fsuper*n) # get the supercell Hamiltonian
        r0 = h.geometry.get_closest_position([0.,0.,0.])[0] # central site
        def get_rkky(d,ii,jj): # define routine
            ri = h.geometry.r[ii]
            rj = h.geometry.r[jj]
            d = np.abs(d) # to the fundamental replica
            dr = d[0]*g0.a1 + d[1]*g0.a2 + d[2]*g0.a3 # shift
            e = h.get_rkky(ri=ri,rj=rj+dr,mode="pm",**kwargs) # get the RKKY
            return e
    elif mode=="LR":
        from .chitk.magneticresponse import rkky_generator
        rkkgen = rkky_generator(h,**kwargs) # function to call 
        def get_rkky(d,ii,jj): # define routine
            e = rkkgen(R=d,ii=ii,jj=jj)
#            e = h.get_rkky(R=np.array(d),ii=ii,jj=jj,
#                             mode="LR",**kwargs) # get the RKKY
            if info: print(d,ii,jj,e)
            return e
    else: raise # not implemented

    eout = [] # output RKKY
    rout = [] # output locations
    for d in g0.neighbor_directions(n): # loop over directions
        d = np.array(d)
        jj = 0
        for ii in range(len(g0.r)): # loop over indexes
                if ii==jj and np.max(np.abs(d))==0: continue # same site
                ri = g0.r[ii] # position
                rj = g0.r[jj] # position
                e = get_rkky(d,ii=ii,jj=jj)
                dr = d[0]*g0.a1 + d[1]*g0.a2 + d[2]*g0.a3
                eout.append(e) # store RKKY
                rout.append(dr+ri-rj) # store location   
    rout = np.array(rout) # convert to array
    eout = np.array(eout) # convert to array
    return np.array([rout[:,0],rout[:,1],rout[:,2],eout]).T # return all
