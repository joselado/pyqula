import numpy as np
import scipy.linalg as lg

def kfun_map(h,nk=50,
               nsuper=1,reciprocal=True,
               operator=None,k0=[0.,0.]):
    """ Calculate a reciprocal space map"""
    if operator is None: raise
    if h.dimensionality!=2: raise  # continue if two dimensional
    hk_gen = h.get_hk_gen() # gets the function to generate h(k)
    kxs = np.linspace(-nsuper,nsuper,nk)+k0[0]  # generate kx
    kys = np.linspace(-nsuper,nsuper,nk)+k0[1]  # generate ky
    kdos = [] # empty list
    kxout = []
    kyout = []
    if reciprocal: R = h.geometry.get_k2K() # get matrix
    else:  R = np.array(np.identity(3)) # get identity
    out = [] # empty list
    kx = []
    ky = []
    for x in kxs:
        for y in kxs:
            print("Doing",x,y)
            r = np.array([x,y,0.]) # real space vectors
            k = np.array(R)@r # change of basis
            hk = hk_gen(k) # get hamiltonian
            o = operator(hk) # make an operation at this kpoint
            out.append(o) # store
            kx.append(x) # store
            ky.append(y) # store
    return kx,ky,out # return result


def conduction_texture(h,n=2,**kwargs):
    """Compute the spin texture in the conduction band"""
    def fun(hk,O=None):
        """Function to call"""
        es,ws = lg.eigh(hk) # diagonalize
        ws = ws.T # transpose
        ws = ws[es>0.] # positive energies
        es = es[es>0.] # positive energies
        m = np.zeros((n,n),dtype=np.complex) # create matrix
        for i in range(n):
            wi = np.conjugate(O@ws[i])
            for j in range(n):
                m[i,j] = ws[j].dot(wi) # compute
        return m # return matrix
    fmx = lambda hk: fun(hk,O=h.get_operator("sx").get_matrix())
    fmy = lambda hk: fun(hk,O=h.get_operator("sy").get_matrix())
    kx,ky,mx = kfun_map(h,operator=fmx,**kwargs) # compute the mx matrix
    kx,ky,my = kfun_map(h,operator=fmy,**kwargs) # compute the my matrix
    # compute trace and determinant
    dmx = [lg.det(mxi).real for mxi in mx]
    dmy = [lg.det(myi).real for myi in my]
    tmx = [np.trace(mxi).real for mxi in mx]
    tmy = [np.trace(myi).real for myi in my]
    # now save the data
    np.savetxt("TRACE_TEXTURE.OUT",np.array([kx,ky,tmx,tmy]).T)
    np.savetxt("DET_TEXTURE.OUT",np.array([kx,ky,dmx,dmy]).T)




