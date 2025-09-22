import numpy as np
from .. import filesystem as fs
from .. import parallel


def multildos(self,es=np.linspace(-2.,2.,20),**kwargs):
    """Compute the ldos at different energies"""
    fs.rmdir("MULTILDOS")
    fs.mkdir("MULTILDOS")
    ds = [] # total DOS
    fo = open("MULTILDOS/MULTILDOS.TXT","w")
    # parallel execution
    out = parallel.pcall(lambda x: self.ldos(energy=x,**kwargs),es)
    for (e,o) in zip(es,out):
        (x,y,d) = o # extract
        ds.append(np.mean(d)) # total DOS
        name0 = "LDOS_"+str(e)+"_.OUT" # name
        name = "MULTILDOS/"+name0
        fo.write(name0+"\n") # name of the file
        np.savetxt(name,np.array([x,y,d]).T) # save data
    np.savetxt("MULTILDOS/DOS.OUT",np.array([es,ds]).T)



def get_ldos(self,energy=0.0,delta=1e-2,nsuper=1,nk=100,
                    write = True,return_rd = False,
                    operator=None,**kwargs):
    """Compute the local density of states"""
    from ..increase_hilbert import full2profile
    h = self.H
    # get the Green's function
    gv = self.get_gf(energy=energy,delta=delta,nsuper=nsuper,nk=nk)
    if operator is not None:
        operator = h.get_operator(operator) # overwrite
        gv = operator*gv # multiply
    ds = [-gv[i,i].imag/np.pi for i in range(gv.shape[0])] # LDOS
    ds = full2profile(h,ds,check=False) # resum if necessary
    ds = np.array(ds) # convert to array
    gs = h.geometry.supercell(nsuper)
    x,y,z,r = gs.x,gs.y,gs.z,gs.r
    if write: np.savetxt("LDOS.OUT",np.array([x,y,ds]).T)
    if return_rd:
        return r,ds
    else:
        return x,y,ds


