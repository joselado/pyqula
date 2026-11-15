import numpy as np

from .. import specialhopping

def get_hamiltonian(self,tij=None,has_spin=True,
                        is_sparse=False,spinful_generator=False,nc=None,
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
        - nc = None, the cells a function or a generator is evaluated on,
          the box |n_k| <= nc (also accepted as cutoff); None is every cell
          that holds a pair of sites within rcut (specialhopping.cells_within)
        - rcut = 5.0, the distance that sets those cells
    A function or a generator on a 1D or 2D lattice is built as multicell,
    on every cell within its range, since the single-cell storage keeps
    only the neighboring cells and a farther hopping was dropped
    (future_development/hopping_range.md). A Hermitian one is evaluated on
    one cell R of each pair R, -R, and the hopping to -R is the adjoint, as
    the single-cell storage did, so it is Hermitian by construction also
    for a function that depends on the absolute position.
    An unknown keyword is an error rather than being ignored.
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
    nc_given = nc is not None # the box of cells asked for, if any
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
    # a function or a generator is built as multicell, on every cell
    # within its range: the single-cell storage of 1d and 2d (inter, tx,
    # ty, txy, txmy) keeps only the neighboring cells, and on the
    # triangular lattice it lost 4 of the 6 second-neighbor bonds
    if self.dimensionality in [1,2] and (tij is not None or
            mgenerator is not None):
        is_multicell = True
    from ..hamiltonians import Hamiltonian
    h = Hamiltonian(self)  # create the object
    h.is_sparse = is_sparse
    h.non_hermitian = non_hermitian # non Hermitian flag
    h.has_spin = has_spin
    h.is_multicell = is_multicell
    if is_multicell:  # workaround for multicell hamiltonians
        from ..multicell import parametric_hopping_hamiltonian
        from ..multicell import parametric_matrix
        # a Hermitian Hamiltonian is evaluated on one cell of each pair R,
        # -R and the other is its adjoint; a non-Hermitian one on both
        cells = hopping_cells(self,nc,nc_given,kwargs,
                half=not non_hermitian)
        if spinful_generator: h.has_spin = True # 2x2 blocks per pair
        if mgenerator is not None: # if mgenerator is given
            h = parametric_matrix(h,fm=mgenerator,cells=cells,
                    spinful_generator=spinful_generator)
        else:
            h = parametric_hopping_hamiltonian(h,fc=tij,cells=cells,
                    spinful_generator=spinful_generator,**kwargs) # add hopping
        if not non_hermitian: add_reverse_cells(h) # the hopping to -R
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
                single_direction=True)
        if mismatch>1e-8:
            import warnings
            stored = ("each bond between cells keeps one of its two "
                "directions and takes the other as its adjoint, and an "
                "intra-cell block that is not Hermitian is kept as it is")
            warnings.warn("the hopping function tij is not reciprocal, "
              +"t(r_i,r_j) differs from t(r_j,r_i)^* by up to "
              +"%.3g relative to the largest hopping, but the " % mismatch
              +"Hamiltonian is built as Hermitian: "+stored+". Pass "
              +"non_hermitian=True for a non-reciprocal hopping",
              stacklevel=3)
    return h # return the object


def is_forward(d):
    """Whether the cell d is the one of the pair d, -d that is evaluated:
    its first nonzero index is positive"""
    for x in d:
        if x!=0: return x>0
    return False # the cell 0


def hopping_cells(g,nc,nc_given,kwargs,half=True):
    """The cells (n1,n2,n3) a hopping function or generator is evaluated
    on: the box |n_k| <= nc, which the multicell builders call cutoff, or
    else every cell with a pair of sites within rcut=5.0. With half, one
    cell of each pair d, -d (is_forward), the other being its adjoint.
    The cell 0 is not in the list, the builders evaluate it apart. Pops
    cutoff from kwargs, and leaves rcut for the function builder, which
    skips a cell with no pair within it"""
    if "cutoff" in kwargs:
        if nc_given:
            raise TypeError("got both nc and cutoff, two names of the box "
                    "of cells of a hopping; pass only one of them")
        nc = kwargs.pop("cutoff")
    if nc is None:
        cells = specialhopping.cells_within(g,kwargs.get("rcut",5.0))
    else:
        from ..neighbor import neighbor_directions
        cells = [tuple(int(x) for x in d) for d in neighbor_directions(g,nc)]
    cells = [c for c in cells if c!=(0,0,0)]
    if half: cells = [c for c in cells if is_forward(c)]
    return cells


def add_reverse_cells(h):
    """Add to a multicell Hamiltonian evaluated on the forward cells
    (is_forward) the hopping to each cell -d, as the adjoint of the
    hopping to d"""
    from ..multicell import Hopping
    from ..algebra import dagger
    out = []
    for t in h.hopping:
        out.append(t)
        out.append(Hopping(d=[-int(x) for x in t.dir],m=dagger(t.m)))
    h.hopping = out


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
    each bond between cells, the forward cells of is_forward, and the
    other taken as its adjoint, so only the directions that were evaluated
    from tij are checked; the derived ones would compare tij with its
    lattice translate instead, which a reciprocal function that is not
    periodic in the lattice would fail"""
    from ..multicell import unit_cell_hoppings
    g = h.geometry
    r = np.array(g.r)
    mats = [h.intra]
    mismatch = _max_abs(h.intra - np.conjugate(h.intra).T)
    if h.is_multicell: pairs = [(t.dir,t.m) for t in h.hopping]
    else: pairs = unit_cell_hoppings(h)
    if single_direction: # the directions evaluated from tij, see above
        pairs = [(d,m) for (d,m) in pairs if is_forward(d)]
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

