import numpy as np

def orthogonality_distribution(vs,n=10):
    """Given a set of eigenvectors, compute the distribution of their
    orthogonality"""
    n0 = len(vs)
    m = [] # output
    for i in range(n0):
        mi = 1.0
        for j in range(n0):
            if i==j: continue
            d = 1.-np.abs(vs[i].dot(np.conjugate(vs[j])))**2 # overlap
            mi *= d
        m.append(mi) # store
    out = np.histogram(m,bins=n,range=(0.,1.)) # return histogram
    dd = out[0]
    dd = dd/np.sum(dd)
    return out[1][0:n],dd

