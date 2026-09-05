import numpy as np
from scipy.sparse import bmat,csc_matrix
from ..algebratk.unitary import make_unitary


def check_and_fix(smatrix,error=1e-7):
    """Given an smatrix as a list, chwck if it is Hermitian,
    and if not fix it"""
#    return smatrix
    # the two leads can have different dimensions, so each diagonal block
    # sets its own size
    n0 = smatrix[0][0].shape[0] # dimension of the first lead
    n1 = smatrix[1][1].shape[0] # dimension of the second lead
    smatrix2 = [[csc_matrix(smatrix[i][j]) for j in range(2)] for i in range(2)]
    smatrix2 = bmat(smatrix2).todense()
    sH = np.conjugate(smatrix2).T
    # S is unitary iff S^-1 == S^H, equivalently S@S^H == I; the latter
    # avoids an explicit matrix inverse (cheaper, better conditioned)
    iden = np.identity(smatrix2.shape[0],dtype=smatrix2.dtype)
    merror = np.max(np.abs(smatrix2@sH-iden)) #  check unitarity
    if merror> error:
#        print("S-matrix is not unitary",error,"Determinant",np.abs(lg.det(sH)))
#        if abs(np.abs(lg.det(sH))-1.0)>1e-2: raise
#    print("S-matrix is unitary",error,"Determinant",np.abs(lg.det(sH)))
#    else: s3 = smatrix2
        smatrix2 = make_unitary(smatrix2)
#        print("Unitarized determinant",np.abs(lg.det(smatrix2)))
    s3 = np.array(smatrix2) # unitarized
    # bmat lays block [i][j] out at rows i, columns j, so the split back
    # has to read it the same way round. It used to read the off-diagonal
    # blocks transposed -- sout[0][1] got s3[n:2n,0:n], which is block
    # [1][0] -- so get_smatrix (check=True is the default) returned the
    # two transmission blocks interchanged. didv and didv_BdG happened not
    # to notice, since two-terminal unitarity makes Tr(t t^dag) equal for
    # the two and the BdG path only reads the diagonal blocks, but
    # get_tmatrix and any caller of the transmission block itself did.
    sout = [[s3[0:n0,0:n0],       s3[0:n0,n0:n0+n1]],
            [s3[n0:n0+n1,0:n0],   s3[n0:n0+n1,n0:n0+n1]]]
    return sout


