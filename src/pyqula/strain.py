
# routines to apply strain to Hamiltonian

def add_strain(h,sr,**kwargs):
    """Take as input a Hamiltonian, return the Hamiltonian
    with applied strain"""
    h.turn_multicell() # multicell Hamiltonian
    if not h.has_eh:
      if not h.has_spin: indg = lambda i: i
      elif h.has_spin: indg = lambda i: i//2
      else: raise
    else: raise
    f = strain_mode(sr,**kwargs) # get the function depending on the mode
    def fm(m,r1,r2): # function to modify hopping
        return strain_matrix(m,r1,r2,indg,f)
    h.modify_hamiltonian_matrices(fm,use_geometry=True) # modify matrices


def strain_mode(sr,sd=None,mode="scalar"):
    """Create a different fucntion depending on the strain mode"""
    if mode=="scalar": # just change the strength of hoppings
        return lambda r,dr: sr(r)
    elif mode=="directional": # direction-dependent change
        return lambda r,dr: sr(dr)
    elif mode=="non_uniform": # spatial and direction-dependent change
        return lambda r,dr: sr(r,dr)
    else: raise



def strain_matrix(m,rs1,rs2,indg,fs): # function to apply strain to a matrix
    mo = m.copy() # copy matrix
    from scipy.sparse import coo_matrix,csc_matrix
    mo = coo_matrix(mo) # turn sparse
    for k in range(len(mo.data)):
        r0 = (rs1[indg(mo.col[k])] + rs2[indg(mo.row[k])])/2.
        dr0 = rs1[indg(mo.col[k])] - rs2[indg(mo.row[k])]
        mo.data[k] = fs(r0,dr0)*mo.data[k]
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


