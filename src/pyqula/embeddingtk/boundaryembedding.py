import numpy as np

def boundary_embedding_gf(EB,selfenergy=1.,energy=0.,delta=1e-4,**kwargs):
    """Given a Hamiltonian, return an Embedded object that has
    selfenergies at the boundaries of the system"""
    if EB.boundary_embedding_generator is None:
        EB.boundary_embedding_generator = selfe_generator(EB.H) # create 
    if callable(selfenergy): selfe = selfenergy(energy)
    else: selfe = selfenergy
    SE = selfe*EB.boundary_embedding_generator # selfenergy
    from .. import algebra
    iden = np.identity(SE.shape[0],dtype=np.complex) # identity
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

