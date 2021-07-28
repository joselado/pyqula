import numpy as np
import scipy.linalg as lg
from . import algebra
from .ldos import spatial_dos,write_ldos
import os
from . import filesystem as fs
from . import parallel

def evolve_local_state(h,i=0,ts=np.linspace(0.,20.,300),
        mode="chi"):
    """Evolve a state that is originally localized in a point"""
    if h.dimensionality!=0: raise # only for 0d
    # create the vector
#    if h.has_spin: raise
    # get the function that does time evolution
    if mode=="green": evol = evolve_green(h,i=i)
    elif mode=="chi": evol = evolve_chi(h,i=i,ts=ts)
    g = h.geometry
    fs.rmdir("MULTITIMEEVOLUTION") # remove folder
    fs.mkdir("MULTITIMEEVOLUTION") # create folder
    fo = open("MULTITIMEEVOLUTION/MULTITIMEEVOLUTION.TXT","w")
    for t in ts: # loop over ts
        out = evol(t) # do the evolution
        print(np.sum(out))
        out = spatial_dos(h,out) # resum if necessary
        name = "TIMEEVOLUTION_T_"+str(t)+"_.OUT" # name
        name2 = "MULTITIMEEVOLUTION/"+name # name
        write_ldos(g.x,g.y,out,output_file=name2) # write the LDOS
        fo.write(name+"\n") # write this file
    fo.close()




def evolve_green(h,i=0):
    """Return the Green function evolution"""
    if h.dimensionality!=0: raise # only for 0d
    # create the vector
    v0 = np.zeros(h.intra.shape[0]) # zero dimensional
    v0[i] = 1.0
    # now do the time evolution
    (es,vs) = algebra.eigh(h.intra) # diagonalize
    vs = vs.T # eigenvectors
    vs = vs[es<0.0,:] # get the eigenvectors
    es = es[es<0.0]
    ws = np.array([np.conjugate(iv[i]) for iv in vs]) # weights
    def evol(t):
        phi = np.exp(1j*es*t) # complex phases
        out2 = np.array([iw*iv*iphi for (iw,iv,iphi) in zip(ws,vs,phi)])
        out = np.zeros(v0.shape[0]) 
        out += np.abs(np.sum(out2,axis=0))**2 # sum over eigenvectors
        return out
    return evol





def evolve_chi(h,i=0,ts=[0]):
    """Function that return the time evolution for chi"""
    from .chi import chargechi_row
    emin = np.min(algebra.eigvalsh(h.intra))
    es = np.linspace(emin,0.,int(max(ts)*2))
    cs = chargechi_row(h,es=es,i=i,delta=abs(emin)/len(es))
    def evol(t):
        out = np.array([c@np.exp(1j*es*t) for c in cs])
        return np.abs(out)
    return evol
