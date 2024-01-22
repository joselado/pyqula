import numpy as np
from .. import algebra
from numba import jit

@jit(nopython=True)
def maximum_wannier_gap(m):
    """Return the maximum wannier gap"""
    x = m[0]
    # find the position of the maximum gap at every t
    fermis = x*0. # maximum gap
    for it in range(len(x)): # loop over times
        dmax = -1 # initialize
        maxgap = -1.0 # maximum gap
        gapangle = ((m[1][it]+0.5)%1)*np.pi # initialize
        for i in range(1,len(m)):
          for j in range(i+1,len(m)):
            for ipi in [0.,1.]:
              ip = np.exp(1j*m[i][it]) # center of wave i
              jp = np.exp(1j*m[j][it]) # center of wave j
              angle = np.angle(ip+jp)+np.pi*ipi # get the angle
              dp = np.exp(1j*angle) # now obtain this middle point gap
              mindis = 4.0 # calculate minimum distance
              for k in range(1,len(m)): # loop over centers
                kp = np.exp(1j*m[k][it]) # center of wave k
                dis = np.abs(dp-kp) # distance between the two points
                if dis<mindis: mindis = dis+0. # update minimum distance
              if mindis>maxgap: # if found a bigger gap
                maxgap = mindis+0. # maximum distance
                gapangle = np.angle(dp) # update of found bigger gap
        fermis[it] = gapangle
    return fermis


from .overlap import uij


def smooth_gauge(w1,w2):
    """Regauge wavefunctions w2 so that they are
    adibatically connected to w1"""
    m = uij(w1,w2) # matrix of wavefunctions
    U, s, V = np.linalg.svd(m, full_matrices=True) # sing val decomp
    R = algebra.dagger(U@V) # rotation matrix
    return smooth_rotation_jit(w2,R)

@jit(nopython=True)
def smooth_rotation_jit(w2,R):
    wnew = w2.copy()*0j # initialize
    wold = w2.copy() # old waves
    for ii in range(R.shape[0]):
      for jj in range(R.shape[0]):
        wnew[ii] += R[jj,ii]*wold[jj]
    return wnew






