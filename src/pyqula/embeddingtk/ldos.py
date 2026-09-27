from .. import filewrite
import numpy as np
from .. import filesystem as fs
from .. import parallel


def multildos(self,es=np.linspace(-2.,2.,20),write=None,**kwargs):
    """Compute the ldos at different energies"""
    write = filewrite.resolve(write,True) # the call, else the global switch
    # parallel execution; each LDOS is an intermediate, never written
    out = parallel.pcall(lambda x: self.ldos(energy=x,write=False,**kwargs),es)
    (x,y) = out[0][0],out[0][1] # the same positions at every energy
    ldos = np.array([o[2] for o in out]) # one map per energy
    if write:
        fs.rmdir("MULTILDOS")
        fs.mkdir("MULTILDOS")
        fo = open("MULTILDOS/MULTILDOS.TXT","w")
        for (e,d) in zip(es,ldos):
            name0 = "LDOS_"+str(e)+"_.OUT" # name
            name = "MULTILDOS/"+name0
            fo.write(name0+"\n") # name of the file
            np.savetxt(name,np.array([x,y,d]).T) # save data
        fo.close()
        ds = [np.mean(d) for d in ldos] # total DOS
        np.savetxt("MULTILDOS/DOS.OUT",np.array([es,ds]).T)
    return x,y,np.array(es),ldos



def get_ldos(self,energy=0.0,delta=1e-2,nsuper=1,nk=100,
                    write=None,return_rd = False,
                    operator=None,**kwargs):
    """Compute the local density of states"""
    write = filewrite.resolve(write,True) # the call, else the global switch
    from ..increase_hilbert import full2profile
    h = self.H
    # get the Green's function
    gv = self.get_gf(energy=energy,delta=delta,nsuper=nsuper,nk=nk)
    if operator is not None:
        operator = h.get_operator(operator) # overwrite
        gv = operator*gv # multiply
    ds = -np.diag(gv).imag/np.pi # LDOS
    ds = full2profile(h,ds,check=False) # resum if necessary
    ds = np.array(ds) # convert to array
    gs = h.geometry
    if self.nsuper is not None: gs = gs.supercell(self.nsuper)
    gs = gs.supercell(nsuper)
    x,y,z,r = gs.x,gs.y,gs.z,gs.r
    if write: np.savetxt("LDOS.OUT",np.array([x,y,ds]).T)
    if return_rd:
        return r,ds
    else:
        return x,y,ds


