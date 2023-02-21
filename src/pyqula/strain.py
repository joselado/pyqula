import numpy as np

# routines to apply strain to Hamiltonian

def add_strain(h,sr,**kwargs):
    """Take as input a Hamiltonian, return the Hamiltonian
    with applied strain"""
    h.turn_multicell() # multicell Hamiltonian
    if not h.has_eh:
      if not h.has_spin: indg = lambda i: i
      elif h.has_spin: indg = lambda i: i//2
      else: raise
    else: 
      if h.has_spin: indg = lambda i: i//4
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



def uniaxial_strain(H,d=np.array([1.,0.,0.]),s=0.,**kwargs):
    """Add uniaxial strain"""
    d = np.array(d) ; d = d/np.sqrt(d.dot(d)) # normalize
    def fun(dr): # function to get the strain
        dr2 = dr.dot(dr) # moduli
        if dr2<1e-4: return 1.0 # do nothing
        dr = dr/np.sqrt(dr2) # normalize
        dr0 = dr.dot(d) # projection on the direction
        fac = np.abs(dr0) # modulation of the strain
        return 1. + 2*(fac - 0.5)*s # change the hopping
    add_strain(H,fun,mode="directional") # add strain



def strain_matrix(m,rs1,rs2,indg,fs): # function to apply strain to a matrix
    mo = m.copy() # copy matrix
    from scipy.sparse import coo_matrix,csc_matrix
    mo = coo_matrix(mo) # turn sparse
    for k in range(len(mo.data)):
        ii = indg(mo.col[k])
        jj = indg(mo.row[k])
        r0 = (rs1[ii] + rs2[jj])/2. # average location 
        dr0 = rs1[ii] - rs2[jj] # distance difference
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


