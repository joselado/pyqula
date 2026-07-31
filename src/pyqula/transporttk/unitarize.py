import numpy as np
from scipy.sparse import bmat,csc_matrix
from ..algebratk.unitary import make_unitary


def check_and_fix(smatrix,error=1e-7):
    """Given an smatrix as a list, chwck if it is Hermitian,
    and if not fix it"""
#    return smatrix
    n = smatrix[0][0].shape[0] # dimension of the matrix
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
    sout = [[s3[0:n,0:n],s3[n:2*n,0:n]],[s3[0:n,n:2*n],s3[n:2*n,n:2*n]]]
    return sout


