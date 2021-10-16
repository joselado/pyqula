
# routines to apply strain to Hamiltonian

def add_strain(h,f):
    """Take as input a Hamiltonian, return the Hamiltonian
    with applied strain"""
    h.turn_multicell() # multicell Hamiltonian
    if not h.has_eh:
      if not h.has_spin: indg = lambda i: i
      elif h.has_spin: indg = lambda i: i//2
      else: raise
    else: raise
    def fm(m,r1,r2): # function to modify hopping
        return strain_matrix(m,r1,r2,indg,f)
    h.modify_hamiltonian_matrices(fm,use_geometry=True) # modify matrices



def strain_matrix(m,rs1,rs2,indg,fs): # function to apply strain to a matrix
    mo = m.copy() # copy matrix
    from scipy.sparse import coo_matrix,csc_matrix
    mo = coo_matrix(mo) # turn sparse
    for k in range(len(mo.data)):
        r0 = (rs1[indg(mo.col[k])] + rs2[indg(mo.row[k])])/2.
        mo.data[k] = fs(r0)*mo.data[k]
    from .algebra import issparse
    if issparse(m): return csc_matrix(mo)
    else: return mo.todense()


def simple_strain_matrix(m,rs1,rs2,indg,fs): # function to apply strain to a matrix
    mo = m.copy() # copy matrix
    for i in range(len(rs1)): # loop over sites
        for j in range(len(rs2)): # loop over sites
            r0 = (rs1[i]+rs2[j])/2.
            fac = fs(r0)
            for ii in indg(i): # generate indexes for site i
                for jj in indg(j): # generate indexes for site j
                    mo[ii,jj] = fac*m[ii,jj] # strain the matrix
    return mo


