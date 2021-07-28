from numba import jit
import numpy as np 



@jit(nopython=True)
def numba_number_first_neighbor(r1,r2):
   """Number of first neighbors"""
   n = 0
   for i in range(len(r1)):
     for j in range(len(r2)):
       ri = r1[i]
       rj = r2[j]
       dr = ri-rj
       dr2 = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]
       if 0.8<dr2<1.2: n +=1
   return n # return neighbors
@jit(nopython=True)
def numba_pairs_first_neighbor(r1,r2,pairs):
   """Return the pairs"""
   n = 0
   for i in range(len(r1)):
     for j in range(len(r2)):
       ri = r1[i]
       rj = r2[j]
       dr = ri-rj
       dr2 = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]
       if 0.8<dr2<1.2:
           pairs[n][0] = i
           pairs[n][1] = j
           n += 1 # increase counter
   return pairs
def find_first_neighbor(r1,r2):
    """Get first neighbors"""
    print("Using numba in find_first_neighbor")
    n = numba_number_first_neighbor(r1,r2) # number of first neighbors
    pairs = np.zeros((n,2),dtype=int) # create the array
    pairs = numba_pairs_first_neighbor(r1,r2,pairs) # get the pairs
    return pairs

