import numpy as np


def bulk2ribbon(hin,n=10):
  """ Create a ribbon hamiltonian object"""
  if hin.dimensionality!=2:
    raise ValueError("bulk2ribbon needs a 2d Hamiltonian")
  hin = hin.get_multicell() # into multicell form
  hr = hin.get_supercell(nsuper=[1,n,1]) # Hamiltonian of the ribbon
  hr.geometry.dimensionality = 1 # one dimensional
  hr.dimensionality = 1 # one dimensional
  out = [] # empty list
  for i in range(len(hr.hopping)): # loop over hoppings
      d = hr.hopping[i].dir # direction
      d = np.array(d) # as array
      if np.max(np.abs(d[1:]))==0: # hopping in x direction
          out.append(hr.hopping[i].copy())
  hr.hopping = out # overwrite the hoppings
  return hr



def bulk2film(hin,n=10):
  """ Create a ribbon hamiltonian object"""
  if hin.dimensionality!=3:
    raise ValueError("bulk2film needs a 3d Hamiltonian")
  hin = hin.get_multicell() # into multicell form
  hr = hin.get_supercell(nsuper=[1,1,n]) # Hamiltonian of the ribbon
  hr.geometry.dimensionality = 2 # two dimensional
  hr.dimensionality = 2 # two dimensional
  out = [] # empty list
  for i in range(len(hr.hopping)): # loop over hoppings
      d = hr.hopping[i].dir # direction
      d = np.array(d) # as array
      if np.max(np.abs(d[2:]))==0: # hopping in x,y direction
          out.append(hr.hopping[i].copy())
  hr.hopping = out # overwrite the hoppings
  return hr






def matrix_supercell(hin,M,sparse=True,ncut=None):
    """The supercell Hamiltonian of h along an integer matrix M, with the
    lattice vectors A_S = M A (rows, M of shape 3x3, the identity beyond the
    dimensionality), for any M, diagonal or not. On a d-dimensional lattice
    M can also be dxd, and it is then padded to 3x3 with 1 along the
    missing directions.

    The geometry is g.get_supercell(M), which records the cell R and the
    unit-cell site of every site. A hopping of the unit cell from orbital a
    of the cell R to orbital b of the cell R+d becomes, for every copy i of
    a, the element between i and the copy j of b in the cell R_j with
    R + d = R_j + D M, in the hopping of the supercell in the direction D.
    The copies are found with the diagonal form P M Q = diag(n), in which
    the cells of the supercell are a grid in R Q (ultracelltk/basis.py).
    Spin and the electron-hole degree of freedom go with each site, as in
    the unit cell. Returns a sparse Hamiltonian unless sparse=False, as
    h.get_supercell does for repetitions. Every hopping of the unit cell
    is kept, so the range cut ncut of the repetitions is refused here"""
    if ncut is not None:
        raise TypeError("ncut="+str(ncut)+" has no meaning for a matrix "
                "supercell, which keeps every hopping of the unit cell; "
                "leave ncut out")
    from scipy.sparse import coo_matrix, csc_matrix
    from ..unfolding import get_supercell_map
    from ..ultracelltk.basis import supercell_shape, pad3
    from ..multicell import Hopping
    h = hin.get_multicell() # a copy
    if h.dimensionality==0: return h
    dim = h.dimensionality
    Mf = np.array(M,dtype=float)
    if Mf.shape not in [(3,3),(dim,dim)]:
        raise ValueError("a supercell matrix of a "+str(dim)+"-dimensional "
                "lattice is "+str(dim)+"x"+str(dim)+" or 3x3, got shape "
                +str(Mf.shape))
    M3f = np.identity(3) # padded with 1 along the missing directions
    M3f[0:Mf.shape[0],0:Mf.shape[1]] = Mf
    Mf = M3f
    # rounded and not truncated, so that a matrix computed in floating
    # point, with 1.9999999999 for 2, is the matrix of those integers
    if not np.all(np.abs(Mf-np.round(Mf))<1e-6): # also refuses a nan
        raise ValueError("the supercell matrix must have integer entries, "
                "got "+str(Mf.tolist()))
    gs = h.geometry.get_supercell(np.round(Mf).astype(int),store_primal=True)
    Ms,replicas,primal = get_supercell_map(gs,h.geometry)
    M3,P,Q,n = supercell_shape(Ms,dim)
    Minv = np.linalg.inv(M3.astype(float))
    N = int(np.prod(n)) # cells of the supercell
    norb = h.intra.shape[0] # orbitals of the unit cell
    nat0 = len(h.geometry.r)
    if norb%nat0!=0 or len(gs.r)!=N*nat0:
        raise ValueError("the supercell of matrix "+str(M3.tolist())+" has "
                +str(len(gs.r))+" sites for "+str(N)+" cells of "+str(nat0)
                +" sites, or the unit cell has no whole number of orbitals "
                "per site")
    f = norb//nat0 # orbitals per site (spin, electron-hole)
    nats = len(gs.r)
    rep3 = np.zeros((nats,3),dtype=int)
    replicas = np.array(replicas,dtype=int).reshape(nats,-1)
    rep3[:,0:replicas.shape[1]] = replicas[:,0:3]
    primal = np.array(primal,dtype=int)
    R = np.repeat(rep3,f,axis=0) # the cell of every orbital
    alpha = np.repeat(primal*f,f) + np.tile(np.arange(f),nats)
    def cell_index(Rs): # the cell inside the supercell, from the grid R Q
        return np.ravel_multi_index(np.mod(Rs@Q,n[None,:]).T,tuple(n))
    table = -np.ones((N,norb),dtype=int) # orbital of (cell, unit-cell orbital)
    table[cell_index(R),alpha] = np.arange(len(alpha))
    if np.any(table<0):
        raise ValueError("the supercell does not hold every orbital of every "
                "one of its cells")
    copies = np.array([np.nonzero(alpha==a)[0] for a in range(norb)]) # (norb,N)
    entries = dict() # direction D of the supercell -> (rows, cols, data)
    for d,m in h.get_multihopping().get_dict().items():
        m = coo_matrix(m)
        if m.nnz==0: continue
        rows = copies[m.row].ravel() # every copy i of a, entry by entry
        Rt = R[rows] + pad3(d).astype(int)[None,:] # the cell R+d
        cols = table[cell_index(Rt),np.repeat(m.col,N)] # the copy j of b
        D = np.round((Rt-R[cols])@Minv).astype(int) # R+d = R_j + D M
        data = np.repeat(m.data,N)
        keys,inv = np.unique(D,axis=0,return_inverse=True)
        inv = np.ravel(inv)
        for ik,key in enumerate(keys):
            s = inv==ik
            e = entries.setdefault(tuple(int(x) for x in key),([],[],[]))
            e[0].append(rows[s]) ; e[1].append(cols[s]) ; e[2].append(data[s])
    dimS = len(alpha)
    def matrix(D):
        rows,cols,data = entries.get(D,([],[],[]))
        if len(rows)==0: m = csc_matrix((dimS,dimS),dtype=np.complex128)
        else: m = csc_matrix(coo_matrix((np.concatenate(data),
                (np.concatenate(rows),np.concatenate(cols))),
                shape=(dimS,dimS)))
        return m if sparse else m.todense()
    hr = h.copy()
    hr.geometry = gs
    hr.is_multicell = True
    hr.is_sparse = bool(sparse)
    hr.intra = matrix((0,0,0))
    hr.hopping = [Hopping(d=np.array(D),m=matrix(D)) for D in entries
            if D!=(0,0,0)]
    return hr
