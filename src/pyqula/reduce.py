# routines to reduce the size of the hamiltonian

def fullsc2minsc(hin):
  """ Reduces the size of the hamiltonian from full superconducting
  to a minimal """
  h = hin.copy() # copy hamiltonian
  n = len(h.intra)/4 # number of spinless orbitals
  m = h.intra
  def reduce_eh(m):
    t = np.matrix([[0j for i in range(2*n)] for j in range(2*n)])
    for i in range(n): # loop over spinless orbitals
      for j in range(n): # loop over spinless orbitals
        t[i,j] = m[2*i,2*j] # electron up
        t[i+n,j+n] = m[2*n+2*i+1,2*n+2*j+1]  # hole down
        t[i,j+n] = m[2*i+1,2*n+2*j+1]  # coupling 
        t[j+n,i] = np.conjugate(t[i,j+n])  # coupling 
    return t
  h.intra = reduce_eh(h.intra) # apply the operator
  if h.dimensionality==1:
    h.inter = reduce_eh(h.inter)
  if h.dimensionality==2:
    h.tx = reduce_eh(h.tx)
    h.ty = reduce_eh(h.ty)
    h.txy = reduce_eh(h.txy)
    h.txmy = reduce_eh(h.txmy)
  return h # return new hamiltonian 
  

