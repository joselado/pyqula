import numpy as np
from scipy.sparse import csc_matrix,identity
import scipy.sparse as sp

dense = False

def block2nambu_matrix(m):
    if dense: return block2nambu_matrix_dense(m)
    else: return block2nambu_matrix_sparse(m)



def block2nambu_matrix_sparse(m):
  '''Reorder a matrix that has electrons and holes, so that
  the order resembles the Nambu spinor in each site.
  The initial matrix is
  H D
  D H
  The output is a set of block matrices for each site in the 
  Nambu form'''
  nr = m.shape[0]//4 # number of positions
  col,row,data = [],[],[]
  for i in range(nr): # electrons
    col = col + [2*i,2*i+1,2*i+2*nr,2*i+1+2*nr]
    row = row + [4*i,4*i+1,4*i+2,4*i+3]
    data = data + [1.,1.,1.,1.]
  R = sp.coo_matrix((data,(row,col)),shape=m.shape,dtype=complex)
  return R.T



def block2nambu_matrix_dense(m):
  '''Reorder a matrix that has electrons and holes, so that
  the order resembles the Nambu spinor in each site.
  The initial matrix is
  H D
  D H
  The output is a set of block matrices for each site in the 
  Nambu form'''
  R = np.matrix(np.zeros(m.shape)) # zero matrix
  nr = m.shape[0]//4 # number of positions
  for i in range(nr): # electrons
    R[2*i,4*i] = 1.0 # up electron
    R[2*i+1,4*i+1] = 1.0 # down electron
    R[2*i+2*nr,4*i+2] = 1.0 # down holes
    R[2*i+1+2*nr,4*i+3] = 1.0 # up holes
  R = csc_matrix(R) # to sparse
  return R


def block2nambu(m):
    R = block2nambu_matrix(m)
    Rh = np.conjugate(R.T)
    return Rh@m@R

reorder = block2nambu

def nambu2block(m):
    R = block2nambu_matrix(m)
    Rh = np.conjugate(R.T)
    return R@m@Rh


