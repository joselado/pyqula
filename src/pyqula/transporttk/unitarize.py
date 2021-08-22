import numpy as np
from scipy.sparse import bmat,csc_matrix
import scipy.linalg as lg
from ..algebratk.unitary import make_unitary


def check_and_fix(smatrix,error=1e-7):
    """Given an smatrix as a list, chwck if it is Hermitian,
    and if not fix it"""
#    return smatrix
    n = smatrix[0][0].shape[0] # dimension of the matrix
    smatrix2 = [[csc_matrix(smatrix[i][j]) for j in range(2)] for i in range(2)]
    smatrix2 = bmat(smatrix2).todense()
    sH = np.conjugate(smatrix2).T
    merror = np.max(np.abs(lg.inv(smatrix2)-sH)) #  check unitarity
    if merror> error:
#        print("S-matrix is not unitary",error,"Determinant",np.abs(lg.det(sH)))
#        if abs(np.abs(lg.det(sH))-1.0)>1e-2: raise
#    print("S-matrix is unitary",error,"Determinant",np.abs(lg.det(sH)))
#    else: s3 = smatrix2
        smatrix2 = make_unitary(smatrix2)
#        print("Unitarized determinant",np.abs(lg.det(smatrix2)))
    s3 = np.matrix(smatrix2) # unitarized
    sout = [[s3[0:n,0:n],s3[n:2*n,0:n]],[s3[0:n,n:2*n],s3[n:2*n,n:2*n]]]
    return sout


