import numpy as np
from .. import algebra
from .. import densitymatrix

def real_space_chern(h,operator=None):
    """Compute the real space Chern number"""
    if h.dimensionality!=0: raise
    X = h.get_operator("xposition").get_matrix()
    Y = h.get_operator("yposition").get_matrix()
#    X = np.diag(h.geometry.x)
#    Y = np.diag(h.geometry.y)
    m = h.get_hk_gen()([0.,0.,0.]) # get the matrix
    P = densitymatrix.occupied_projector(m).T # projector on occupied states
    Q = np.identity(P.shape[0]) - P # projector on unoccupied states
#    P = np.conjugate(P.T)
    r = h.geometry.r
    dr = np.array([ri.dot(ri) for ri in r])
    rcut = np.max(dr)/3.
    scale = np.pi*rcut/len(dr[dr<rcut]) # scaling for the Chern number
#    Ph = np.conjugate(P.T) # dagger
    A,B = P@X@P,P@Y@P # define the two operators
    if operator is not None: # in case there is a certain projector
        op = operator.get_matrix()
        C1 = A@op@B - B@A@op # compute the commutator
        C2 = A@B@op - B@op@A # compute the commutator
        C = (C1+C2)/2. # average
    elif operator is None: C = A@B - B@A # compute the commutator
    else: raise
    C = np.pi*2*np.diagonal(C).imag # diagonal part
    C = C/scale # normalize
#    print(P - P@P)
    C = h.full2profile(C) # resum if necessary
    h.geometry.write_profile(C,name="REAL_SPACE_CHERN.OUT")






