import numpy as np

# Pfaffian of a dense antisymmetric matrix by the Parlett-Reid
# tridiagonalization with pivoting, following pfaffian_LTL of pfapack
# (M. Wimmer, MIT license, arXiv:1102.3440)


def pfaffian(A,tol=1e-8):
    """Pfaffian of an antisymmetric (complex or real) matrix A

    The elimination brings A to tridiagonal form by congruences with unit
    lower-triangular matrices, whose Pfaffian is the product of the
    superdiagonal entries it leaves, and each row and column interchange
    of the pivoting flips the sign."""
    A = np.array(A,dtype=complex) # a copy, modified below
    n = A.shape[0]
    if A.shape!=(n,n):
        raise ValueError("the Pfaffian needs a square matrix, and this one "
          +"has shape "+str(A.shape))
    if np.max(np.abs(A+A.T),initial=0.)>tol*max(1.,np.max(np.abs(A),
            initial=0.)):
        raise ValueError("the Pfaffian needs an antisymmetric matrix, and "
          +"A+A^T has entries of size "+str(np.max(np.abs(A+A.T))))
    if n%2==1: return 0.
    pf = 1.
    for k in range(0,n-1,2):
        kp = k+1+np.argmax(np.abs(A[k+1:,k])) # largest entry below A[k,k]
        if kp!=k+1: # interchange rows and columns k+1 and kp
            A[[k+1,kp],k:] = A[[kp,k+1],k:]
            A[k:,[k+1,kp]] = A[k:,[kp,k+1]]
            pf = -pf
        if A[k+1,k]==0.: return 0. # a zero pivot means a zero Pfaffian
        pf = pf*A[k,k+1]
        if k+2<n: # eliminate the rest of row and column k
            tau = A[k,k+2:]/A[k,k+1]
            A[k+2:,k+2:] += np.outer(tau,A[k+2:,k+1])
            A[k+2:,k+2:] -= np.outer(A[k+2:,k+1],tau)
    return pf
