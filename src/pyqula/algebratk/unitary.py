import numpy as np
import scipy.linalg as lg

def make_unitary(M):
    """Given a non-unitary matrix, make it unitary"""
    (u,s,vh) = lg.svd(M) # perform the SVD
    s = s/np.abs(s) # normalize singular values
    M = u@np.diag(s)@vh # return the unitarized matrix
    return M
#    return M/np.abs(lg.det(M))
#    (evals,evecs) = lg.eig(M) # eigenvals and eigenvecs
#    print(np.abs(evals))
##    evecs = np.conjugate(evecs).T # change of basis
#    m2 = np.array([[0.0j for i in evals] for j in evals]) # create matrix
#    for i in range(len(evals)):  
#        m2[i,i] = evals[i]/np.abs(evals[i])
##    m2 = evecs.H * m2 * evecs  # change of basis
#    m3 = np.conjugate(evecs).T @ m2 @ evecs  # change of basis
#    return m3 # return matrix

