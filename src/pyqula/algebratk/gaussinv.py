import numpy as np
from scipy.linalg import lu_factor, lu_solve


def gauss_inverse(m,i=0,j=0,test=False):
    """ Calculates the inverse of a block diagonal
        matrix. This uses brute force inversion,
        so very demanding for large matrices."""
#    i,j =0,0
    nb = len(m) # number of blocks
    ca = [None for ii in range(nb)]
    ua = [None for ii in range(nb-1)]
    da = [None for ii in range(nb-1)]
    for ii in range(nb): # diagonal part
      ca[ii] = m[ii][ii] 
    for ii in range(nb-1): 
      ua[ii] = m[ii][ii+1]
      da[ii] = m[ii+1][ii] 
    # in case you use the -1 notation of python
    if i<0: i += nb
    if j<0: j += nb
    # now call the actual fortran routine
    nm = nb # number of blocks
    n = ua[0].shape[0] # dimension of the matrix
#    try:
#        raise
#        from ..fortran.gauss_inv import gauss_inv
#        mout = gauss_inv.gauss_inv(ca,da,ua,i+1,j+1)
#    except:
    from ..green import block_inverse
#        print("Fortran routine was not compiled, using full version")
#        mout = block_inverse(m,i=i,j=j)
    mout = inv_block(ca,da,ua,i,j)
    mout = np.array(mout)
    test = False # test if the inversion worked
    if test: # check whether the inversion worked
        from ..green import block_inverse
        m2 = block_inverse(m,i=i,j=j)
        error = np.round(np.max(np.abs(mout - m2)),4)
        print("Error",error)
    return mout






def inv_block(ca, da, ua, i, j):
    """Gauss inversion, adapted from the fortran function"""
    nm = len(ca)
    n = ca[0].shape[0]
    cl = np.zeros((n, n), dtype=np.complex_)
    cr = np.zeros((n, n), dtype=np.complex_)
    dl = np.zeros((n, n), dtype=np.complex_)
    dr = np.zeros((n, n), dtype=np.complex_)

    for i1 in range(n):
        cl[i1, i1] = 1.0
        cr[i1, i1] = 1.0

    # Calculate dl[i]
    for i1 in range(i+1):
        if i1 != 0:
            a = da[i1-1].copy()
            hm1 = multiply(a, dl)
            a = ua[i1-1].copy()
            dl = multiply(hm1, a)
        a = ca[i1].copy()
        dl = a - dl
        dl = inverse(dl)

        if (i > j) and (i1 <= (i-1)) and (i1 >= j):
            a = da[i1].copy()
            hm1 = multiply(a, dl)
            hm2 = multiply(hm1, cl)
            cl = -hm2.copy()

    for i4 in range(i, nm):
        i1 = (nm -1) + i - i4 
        if i1 != (nm-1):
            a = ua[i1].copy()
            hm1 = multiply(a, dr)
            a = da[i1].copy()
            dr = multiply(hm1, a)
        a = ca[i1].copy()
        dr = a - dr
        dr = inverse(dr)

        if i < j and i1 >= (i+1) and i1 <= j:
            a = ua[i1-1].copy()
            hm1 = multiply(a, dr)
            hm2 = multiply(hm1, cr)
            cr = -hm2.copy()

    dr = inverse(dr)
    dl = inverse(dl)

    a = ca[i].copy() # assign
    hm1=-a+dl+dr
    hm1 = inverse(hm1)
    if i==j: g = hm1.copy()
    elif i<j: g = multiply(hm1,cr)
    elif j<i: g = multiply(hm1,cl)
    return g



def multiply(a, b):
    return a@b

def inverse(a):
    from scipy.linalg import inv
    return inv(a)
