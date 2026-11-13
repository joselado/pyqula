import numpy as np

def boundary_embedding_gf(EB,selfenergy=1.,energy=0.,delta=1e-4,**kwargs):
    """Green's function of EB.H with a selfenergy at the boundaries of
    the system, -1j*selfenergy on the edge sites.

    That selfenergy is retarded, so it belongs to delta>0. For delta<0
    (the lower contour of embeddingtk.embedded.get_dm) the advanced
    Green's function is returned, G(z) = [G(z^*)]^dagger, whose poles
    stay in the upper half plane; keeping -1j*selfenergy there instead
    would put them inside the lower contour."""
    if delta<0.: # advanced branch, the continuation below the real axis
        gv = boundary_embedding_gf(EB,selfenergy=selfenergy,
                energy=np.conjugate(energy),delta=-delta,**kwargs)
        return np.conjugate(gv).T
    if EB.boundary_embedding_generator is None:
        EB.boundary_embedding_generator = selfe_generator(EB.H) # create 
    if callable(selfenergy): selfe = selfenergy(energy)
    else: selfe = selfenergy
    SE = selfe*EB.boundary_embedding_generator # selfenergy
    from .. import algebra
    iden = np.identity(SE.shape[0],dtype=np.complex128) # identity
    emat = iden*(energy + delta*1j) # energy matrix
    gv = algebra.inv(emat - EB.H.intra - SE)   # Defective Green function
    return gv




def selfe_generator(H):
    """Return the matrix with the selfenergy"""
    from ..potentials import edge_potential
    fe = edge_potential(H.geometry) # get the edge potential
    HE = H*0. # make a copy
    HE.add_onsite(fe) # generate a dummy onsite energy
    SE = -1j*HE.intra # matrix acting as the selfenergy
    return SE # return the basic matrix

