from .. import filewrite
import numpy as np
from .. import filesystem as fs
from .. import parallel
from .. import algebra


def multildos(self,es=np.linspace(-2.,2.,20),energies=None,write=None,
        **kwargs):
    """Compute the ldos at different energies, given as es or, with the
    name of h.get_multildos and of Embedding.multidos, as energies"""
    write = filewrite.resolve(write,True) # the call, else the global switch
    if energies is not None: es = energies # overwrite
    (x,y,ldos) = ldos_sweep(self,es,**kwargs) # one map per energy
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


def ldos_sweep(self,es,delta=1e-2,nsuper=1,nk=100,operator=None,
                 **kwargs):
    """The maps of get_ldos at every energy of es, as the positions
    x and y and a (len(es),number of sites) array, the maps get_ldos
    returns energy by energy.

    With gf_mode="triangle" (a single 3d cell, small) the bulk selfenergy
    of the whole sweep comes from one diagonalization of the nk^3 mesh
    (embedding.get_gf_exact_batch) instead of one per energy, and the
    Dyson step is done energy by energy. Every other mode goes energy by
    energy and keeps only the map of each energy, since the Green's
    functions of a large supercell at every energy of a sweep would not
    fit in memory."""
    op = None
    if operator is not None: op = operator_matrix(self,operator,nsuper=nsuper)
    if self.selfenergy is None and self.gf_mode=="triangle":
        from ..embedding import get_gf_exact_batch
        gvs = get_gf_exact_batch(self,es,delta=delta,nsuper=nsuper,
                nk=nk)
        ldos = [gf2profile(self,gv,op=op) for gv in gvs]
    else: # energy by energy
        def f(e):
            gv = self.get_gf(energy=e,delta=delta,nsuper=nsuper,nk=nk)
            return gf2profile(self,gv,op=op)
        ldos = parallel.pcall(f,es)
    gs = embedded_geometry(self,nsuper=nsuper)
    return gs.x,gs.y,np.array(ldos)



def get_ldos(self,energy=0.0,delta=1e-2,nsuper=1,nk=100,
                    write=None,return_rd = False,
                    operator=None,**kwargs):
    """Compute the local density of states

    operator: None, or an operator spec (a name, a matrix, an Operator,
        a potential). The map is then the local matrix element
        Re<psi|P_i A|psi> of the states on site i, read off the Green's
        function as -Im diag(G A + A G)/(2 pi) summed over the orbitals
        of the site (ldos.green2ldos), the convention of h.get_ldos in
        every mode. For an operator local to each site (a spin
        component, the sublattice) it is the same as -Im diag(A G)/pi;
        for a bond operator, a hopping or a current, each end of a bond
        gets half of its weight. A name or a potential is built on the
        embedded cell, the supercell when nsuper>1; a matrix has to have
        the dimension of that cell."""
    write = filewrite.resolve(write,True) # the call, else the global switch
    # the Green's function, unweighted: the operator enters below
    gv = self.get_gf(energy=energy,delta=delta,nsuper=nsuper,nk=nk)
    op = None
    if operator is not None: op = operator_matrix(self,operator,nsuper=nsuper)
    ds = gf2profile(self,gv,op=op) # LDOS
    gs = embedded_geometry(self,nsuper=nsuper)
    x,y,z,r = gs.x,gs.y,gs.z,gs.r
    if write: np.savetxt("LDOS.OUT",np.array([x,y,ds]).T)
    if return_rd:
        return r,ds
    else:
        return x,y,ds


def gf2profile(self,gv,op=None):
    """The LDOS of each site from the Green's function gv of the embedded
    cell, resolved with the dense matrix op as -Im diag(G A + A G)/(2 pi)
    (ldos.green2ldos), summed over the orbitals of each site"""
    from ..ldos import green2ldos
    from ..increase_hilbert import full2profile
    ds = green2ldos(algebra.todense(gv),op=op) # LDOS of each orbital
    return np.array(full2profile(self.H,ds,check=False)) # resum orbitals


def embedded_hamiltonian(self,nsuper=1):
    """The Hamiltonian of the cell on which get_gf returns the Green's
    function: self.H, enlarged by the supercell of a defect given in a
    supercell (Embedding(h,m=...,nsuper=...)) and by the outer supercell
    nsuper of get_gf. With a boundary selfenergy the Green's function is
    the one of the finite system self.H itself."""
    h = self.H
    if self.selfenergy is not None: return h # see boundary_embedding_gf
    if self.nsuper is not None: h = h.supercell(self.nsuper)
    if nsuper!=1: h = h.supercell(nsuper)
    return h


def embedded_geometry(self,nsuper=1):
    """The geometry of the cell of get_ldos, the one of
    embedded_hamiltonian"""
    gs = self.H.geometry
    if self.nsuper is not None: gs = gs.supercell(self.nsuper)
    return gs.supercell(nsuper)


def operator_matrix(self,operator,nsuper=1):
    """Dense matrix of an operator on the cell of get_gf. A name, a
    potential or a callable is built on embedded_hamiltonian, so it has
    the dimension of the supercell when nsuper>1"""
    h = embedded_hamiltonian(self,nsuper=nsuper)
    op = h.get_operator(operator).get_matrix(required=False)
    if op is None:
        raise NotImplementedError("the LDOS and the Green's function of an "
                "embedding are weighted after the integration over the "
                "Brillouin zone, so they need an operator with a matrix; "
                "this one is defined only by its action on a state, as a "
                "k-dependent operator such as the valley one is")
    op = algebra.todense(op)
    n = h.intra.shape[0] # dimension of the embedded cell
    if op.shape!=(n,n):
        raise ValueError("the operator matrix has dimension "
                +str(op.shape[0])+" and the embedded cell "+str(n)
                +"; pass a matrix of the embedded cell, or a name or a "
                "potential, which are built on it")
    return op
