import numpy as np

def kdos(self,kpath=None,energies=None,
           write=True,**kwargs):
    """Compute momentum-resolved spectral function"""
    def fun(k,e):
        if self.dimensionality==2: # 2D heterostructure
            HT1 = self.generate(k) # generate heterostructure
            return HT1.get_coupled_central_dos(energy=e,**kwargs)
        else: raise # not implemented
    if kpath is None: kpath = np.linspace(0.,1.,40)
    if energies is None: energies = np.linspace(-1.0,1.,40)
    energies = np.array(energies)
    from ..parallel import pcall
    kout,eout,dout = [],[],[]
    ds = pcall(lambda k: [fun(k,e) for e in energies],kpath) # call in parallel
    for (k,d) in zip(kpath,ds): # loop over kpoints
        kout = np.concatenate([kout,energies*0.+k]) # store kpoint
        eout = np.concatenate([eout,energies]) # store energies
        dout = np.concatenate([dout,d]) # store DOS
    if write:
        np.savetxt("KDOS.OUT",np.array([kout,eout,dout]).T)
    return (kout,eout,dout)

