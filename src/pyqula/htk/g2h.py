import numpy as np

from .. import specialhopping

def get_hamiltonian(self,tij=None,has_spin=True,
                        is_sparse=False,spinful_generator=False,nc=2,
                        non_hermitian = False,
                        is_multicell=False,mgenerator=None,**kwargs):
    """ Create the hamiltonian for this geometry. By default, it assumes
    first neighbor hopping
        - has_spin = True, whether if to include spin degree of freedom
        - is_sparse = False, use sparse representation
        - tij = None, either a function returning the spatially dependent
          hopping, a specialhopping.HoppingGenerator, or a list of the
          1st,2nd,3rd... neighbor hoppings
        - is_multicell = False, store the hoppings as a multicell dictionary
        - mgenerator = None, generator of the full hopping matrix
        - nc = 2, neighbor cutoff for the multicell construction
    Any remaining keyword is forwarded to the multicell hopping builders
    (cutoff, rcut); an unknown one is an error rather than being ignored.
    """
    ### Perform some initial sanity checks
    # `fun` is what this argument used to be called before it was renamed
    # to `tij`. It is still spelled that way in several places (spinwaves,
    # surface_TI, operators.get_sublattice_hopping_generator, examples),
    # and until now it was silently dropped -- so those built a plain
    # first-neighbor Hamiltonian instead of the one their hopping function
    # describes.
    if "fun" in kwargs:
        if tij is not None:
            raise TypeError("got both 'tij' and its old name 'fun'; pass "
              +"only one of them")
        tij = kwargs.pop("fun")
    # anything not consumed here is only meaningful for the multicell
    # builders below; unknown keywords used to be dropped in silence, so a
    # misspelled has_spin/is_sparse quietly gave the default Hamiltonian
    forwardable = ["cutoff","rcut"] # accepted by the multicell builders
    unknown = [k for k in kwargs if k not in forwardable]
    if len(unknown)>0:
        raise TypeError("get_hamiltonian() got unexpected keyword "
          +"argument(s) "+str(sorted(unknown))+"; the accepted ones are "
          +str(["tij","has_spin","is_sparse","spinful_generator","nc",
                "non_hermitian","is_multicell","mgenerator"]+forwardable))
    ## in case tij is an iterable with hoppings
    from ..checkclass import is_iterable
    if is_iterable(tij): # tij is an iterable
        ts = tij # overwrite the 1st, 2nd etc hoppings
        mgenerator = specialhopping.neighbor_hopping_matrix(self,ts)
        is_multicell = True # overwrite
        nc = 2*len(ts) + 1 # overwrite
        tij = None # set tij to None
    if type(tij)==specialhopping.HoppingGenerator: # for hopping generator
        mgenerator = tij.f # store the function
        tij = None # and set to None
    if self.dimensionality==3: is_multicell=True
    # the single-cell storage of 1d and 2d Hamiltonians (inter, tx, ty...)
    # keeps one direction of each bond and takes the other as its adjoint,
    # which turns a non-reciprocal hopping (t_ij != t_ji^*) into a
    # reciprocal one, so a non-Hermitian hopping is built as multicell,
    # which evaluates both directions
    if non_hermitian and self.dimensionality in [1,2] \
            and (tij is not None or mgenerator is not None):
        is_multicell = True
    from ..hamiltonians import Hamiltonian
    h = Hamiltonian(self)  # create the object
    h.is_sparse = is_sparse
    h.non_hermitian = non_hermitian # non Hermitian flag
    h.has_spin = has_spin
    h.is_multicell = is_multicell
    if is_multicell:  # workaround for multicell hamiltonians
        from ..multicell import parametric_hopping_hamiltonian
        if mgenerator is not None: # if mgenerator is given
            from ..multicell import parametric_matrix # not implemented
            h = parametric_matrix(h,fm=mgenerator,cutoff=nc,**kwargs)
        else:
            if spinful_generator: h.has_spin = True # 2x2 blocks per pair
            h = parametric_hopping_hamiltonian(h,fc=tij,
                    spinful_generator=spinful_generator,**kwargs) # add hopping
#        return h
    else: # non multicell
        if tij is None and mgenerator is None: # no function given
          h.first_neighbors()  # create first neighbor hopping
        else: # function or mgenerator given
          if h.dimensionality<3:
            from ..hamiltonians import generate_parametric_hopping
            h = generate_parametric_hopping(h,f=tij,
                      spinful_generator=spinful_generator,
                      mgenerator=mgenerator) # add hopping
          elif h.dimensionality==3:
            if mgenerator is not None:
                raise NotImplementedError("a matrix generator is not "
                        "supported for 3d Hamiltonians, pass a hopping "
                        "function as tij instead")
            from ..multicell import parametric_hopping_hamiltonian
            h = parametric_hopping_hamiltonian(h,fc=tij,**kwargs) # add hopping
    # ensure right sparsity structure
    if not is_sparse: 
        h = h.get_dense() # dense Hamiltonian
    else:
        h.turn_sparse() # sparse Hamiltonian
    if tij is not None and not non_hermitian: # a Hermitian one from a function
        mismatch = reciprocity_mismatch(h,tij,spinful_generator,
                single_direction=not is_multicell)
        if mismatch>1e-8:
            import warnings
            if is_multicell: stored = ("both directions of each bond are "
                "stored, so the Bloch Hamiltonian is not Hermitian although "
                "it is treated as one")
            else: stored = ("each bond between cells keeps one of its two "
                "directions and takes the other as its adjoint, and an "
                "intra-cell block that is not Hermitian is kept as it is")
            warnings.warn("the hopping function tij is not reciprocal, "
              +"t(r_i,r_j) differs from t(r_j,r_i)^* by up to "
              +"%.3g relative to the largest hopping, but the " % mismatch
              +"Hamiltonian is built as Hermitian: "+stored+". Pass "
              +"non_hermitian=True for a non-reciprocal hopping",
              stacklevel=3)
    return h # return the object


def _max_abs(m):
    """Largest absolute value of a dense or sparse matrix, 0 if empty"""
    import scipy.sparse as sp
    if sp.issparse(m): m = m.tocoo().data
    m = np.abs(np.asarray(m))
    return float(np.max(m)) if m.size>0 else 0.


def _entries(m):
    """(rows,cols,values) of the nonzero entries of a dense or sparse matrix"""
    import scipy.sparse as sp
    if sp.issparse(m):
        m = m.tocoo()
        return m.row,m.col,m.data
    m = np.asarray(m)
    rows,cols = np.nonzero(m)
    return rows,cols,m[rows,cols]


def reciprocity_mismatch(h,tij,spinful_generator=False,
        single_direction=False):
    """Largest |t_ij - t_ji^*| of a Hamiltonian built as Hermitian from the
    hopping function tij, relative to its largest hopping.

    The intra-cell block holds both directions of each of its bonds, so it
    is compared with its adjoint. For the bonds between cells tij is
    evaluated in the opposite direction at every nonzero bond, which costs
    as many calls as there are bonds rather than a second build.

    single_direction: the Hamiltonian was built with a single direction of
    each bond between cells (inter, tx, ty, txy, txmy) and the other taken
    as its adjoint, also after turn_spinful has made it multicell, so only
    the directions that were evaluated from tij are checked; the derived
    ones would compare tij with its lattice translate instead, which a
    reciprocal function that is not periodic in the lattice would fail"""
    from ..multicell import unit_cell_hoppings
    g = h.geometry
    r = np.array(g.r)
    mats = [h.intra]
    mismatch = _max_abs(h.intra - np.conjugate(h.intra).T)
    if h.is_multicell: pairs = [(t.dir,t.m) for t in h.hopping]
    else: pairs = unit_cell_hoppings(h)
    if single_direction: # the directions stored from tij, see above
        stored = [(1,0,0),(0,1,0),(1,1,0),(1,-1,0)]
        pairs = [(d,m) for (d,m) in pairs
                    if tuple(int(x) for x in d) in stored]
    spinful = spinful_generator or h.has_spin # two entries per site
    for d,m in pairs:
        mats.append(m)
        R = d[0]*np.array(g.a1) + d[1]*np.array(g.a2) + d[2]*np.array(g.a3)
        cache = dict() # tij in the opposite direction, per pair of sites
        rows,cols,vals = _entries(m)
        for a,b,v in zip(rows,cols,vals):
            i,j = (a//2,b//2) if spinful else (a,b)
            if (i,j) not in cache: cache[(i,j)] = tij(r[j]+R,r[i])
            t = cache[(i,j)] # hopping from r_j+R back to r_i
            if spinful_generator: t = np.asarray(t)[b%2,a%2]
            mismatch = max(mismatch,abs(t - np.conjugate(v)))
    scale = max([_max_abs(m) for m in mats]+[0.])
    return mismatch/scale if scale>0. else 0.

