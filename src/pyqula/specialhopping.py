import numpy as np
from scipy.sparse import csc_matrix
from numba import jit


def obj2callable(a):
    if callable(a): return a # input is a function
    else: return lambda x: a # input is a number



def twisted(cutoff=5.0,ti=0.3,lambi=8.0,t=1.0,
        lamb=12.0,dl=3.0,lambz=10.0,b=0.0,phi=0.0):
    """Hopping for twisted bilayer graphene"""
    ti = obj2callable(ti) # convert to callable
    cutoff2 = cutoff**2 # cutoff in distance
    def fun(r1,r2):
        rr = (r1-r2) # distance
        rm  = (r1+r2)/2. # average location
        rr = rr.dot(rr) # distance
        if rr>cutoff2: return 0.0 # too far
        if rr<0.001: return 0.0 # same atom
        dx = r1[0]-r2[0]
        dy = r1[1]-r2[1]
        dz = r1[2]-r2[2]
        r = np.sqrt(rr)
        if (r-1.0)<-0.1:
            raise ValueError("two sites are closer than the first-neighbor "
                    "distance this hopping is normalized to")
        out = -t*(dx*dx + dy*dy)/rr*np.exp(-lamb*(r-1.0))*np.exp(-lambz*dz*dz)
        # interlayer hopping
        out += -ti(rm)*(dz*dz)/rr*np.exp(-lambi*(r-dl))
        #### fix for magnetic field
  #      cphi = np.cos(phi*np.pi)
  #      sphi = np.sin(phi*np.pi)
  #      r = (r1+r2)/2.
  #      dr = r1-r2
  #      p = 2*r[2]*(dr[0]*sphi - dr[1]*cphi)
  #      out *= np.exp(1j*b*p)
        #####
        return out
    return fun


def real_graphene():
    return twisted(ti=0.15,lamb=0.5)



def twisted_matrix(cutoff=5.0,ti=0.3,lambi=8.0,mint=1e-5,
        t=1.0,lamb=12.0,dl=3.0,lambz=10.0,**kwargs):
  """Function capable of returning the hopping matrix
  for twisted bilayer graphene"""
  if True:
      if callable(ti): # workaround for callable interlayer hopping
          tij = twisted(cutoff=cutoff,ti=ti,t=t,
                          lambi=lambi,lamb=lamb,dl=dl,lambz=lambz)
          return entry2matrix(tij)
      else: # conventional JIT function
          return twisted_matrix_python(cutoff=cutoff,ti=ti,mint=mint,
                                  t=t,
                                  lambi=lambi,lamb=lamb,lambz=lambz,
                                  dl=dl,**kwargs)


def multilayer(ti=0.3,dz=3.0):
    """Return hopping for a multilayer"""
    def fhop(ri,rj):
      """Function to compute the hopping"""
      dr = ri-rj ; dr2 = dr.dot(dr) # distance
      if abs(1.0-dr2)<0.01: return 1.0 # first neighbors
      # interlayer hopping (distance between the layers is 3)
      if abs(dz**2-dr2)<0.01 and abs(dz-abs(dr[2]))<0.01: return ti
      return 0.0 # else
    return fhop

def entry2matrix(f):
    """Transform a function that returns a hopping into a function
    that returns the hopping matrix"""
    try:
        fnum = jit(f,nopython=True) # jit function
        def fout(rs1,rs2):
            n1 = len(rs1)
            n2 = len(rs2)
            out = np.zeros((n1,n2),dtype=np.complex128)
            for i in range(n1):
                r1 = rs1[i]
                for j in range(n2):
                    r2 = rs2[j]
                    out[i,j] = fnum(r1,r2)
            return out
        foutjit = jit(fout,nopython=True)
        r0 = np.zeros((1,3)) # dummy input to force compilation now,
        foutjit(r0,r0) # since jit is lazy and only compiles on first call
        return foutjit
    except Exception:
        print("Warning, hopping cannot be jitd")
        def fout(rs1,rs2):
            return np.array([[f(r1,r2) for r1 in rs1] for r2 in rs2]).T
    return fout


def phase_C3_matrix(*args,**kwargs):
    f = phase_C3(*args,**kwargs)
    return entry2matrix(f) # return the matrix


def phase_C3(g,phi=0.5,t=1.0,d=1.0):
    """Create a function that computes hoppings that alternate
    between +phi and -phi every 60 degrees"""
    if len(g.r)==1:
      g = g.get_supercell(2) # create a supercell
    ds = g.get_connections()
    i = 0 # first site
    j = ds[i][0] # connected site
    dr = g.r[i] - g.r[j] # distance between sites
    z = dr[0] + 1j*dr[1] # comple vector
    z0 = np.exp(1j*np.pi*2./3.) # rotation
    zs = [z,z*z0,z*z0**2] # the three vectors
    def fun(r1,r2):
        """Function to compute hoppings"""
        dr = r1-r2
        if (d-0.01)<dr.dot(dr)<(d+0.01): # first neighbors
            zi = dr[0]+1j*dr[1]
            for zj in zs: # one of the three directions
                dd = np.abs(zi/zj-1.0)
                if dd<1e-2: 
                    return t*np.exp(1j*phi*np.pi)
            else: return np.exp(-1j*phi*np.pi)
        return 0.0
    return fun


def neighbor_hopping_matrix(g,vs):
    """Return a hopping matrix for the N first neighbors"""
    ds = g.neighbor_distances(n=len(vs)) # get the different distances
    ds = ds[0:len(vs)] # take only these
    return distance_hopping_matrix(vs,ds)




def distance_hopping_matrix(vs,ds):
    """Return a hopping that to the 1-th neighbor is vs"""
    vs = np.array(vs)
    ds = np.array(ds)
    def mgenerator(r1,r2):
        r1 = np.array(r1)
        r2 = np.array(r2)
        n = len(r1)
        out = np.zeros((n,n),dtype=np.complex128) # output
        return distance_hopping_matrix_jit(r1,r2,vs,ds*ds,out) 
    return mgenerator

def _show_coupling(x):
    """A coupling as text: a real one as a number"""
    return "%g"%x.real if x.imag==0. else str(x)


def _pair_text(d,i,j):
    return ("from site %d to site %d of the cell (%d,%d,%d)"
            % (i,j,d[0],d[1],d[2]))


def require_pair_interaction(v,name="Vr",tol=1e-8):
    """Raise unless v is an interaction that a pair of sites can have: every
    entry finite, real, and the same from the two ends of the pair
    (require_reciprocal_interaction). v is a {direction: matrix} dictionary
    between the sites, dense or sparse, and name is the keyword the caller
    gave the interaction with (Vr, Jr), for the message.

    The order matters. An infinite entry (a Coulomb tail without a soft
    core at zero distance) would be the scale of the two relative
    tolerances and switch them off, and a NaN hides the largest mismatch.
    The coupling of n_i n_j or of S_i.S_j is a real number, and for a real
    v the transpose of the reciprocity check is the right comparison: with
    a complex one a purely imaginary coupling would pass it, and a
    Hermitian one would fail it for the wrong reason."""
    from scipy.sparse import csr_matrix
    v = {tuple(int(x) for x in d): csr_matrix(m).tocoo() for d,m in v.items()}
    for d,m in v.items():
        bad = np.nonzero(~np.isfinite(m.data))[0]
        if len(bad)>0:
            k = bad[0]
            raise ValueError(("%s is not finite: %s = %s %s; a function "
                "that diverges at zero distance must return a number there "
                "(a soft core, or zero for a coupling between different "
                "sites only)") % (name,name,_show_coupling(m.data[k]),
                _pair_text(d,m.row[k],m.col[k])))
    scale = max([np.max(np.abs(m.data)) for m in v.values() if m.nnz>0]+[0.])
    for d,m in v.items():
        if m.nnz==0: continue
        k = np.argmax(np.abs(m.data.imag))
        if np.abs(m.data[k].imag)>tol*scale:
            raise ValueError(("%s must be real: %s = %s %s; the coupling "
                "of n_i n_j or of S_i.S_j between two sites is a real "
                "number") % (name,name,_show_coupling(m.data[k]),
                _pair_text(d,m.row[k],m.col[k])))
    require_reciprocal_interaction(v,name=name,tol=tol)


def require_reciprocal_interaction(v,name="Vr",tol=1e-8):
    """Raise if the interaction of some pair of sites depends on the end it
    is seen from, v[d][i,j] != v[-d][j,i], beyond tol times the largest
    entry. v is a {direction: matrix} dictionary between the sites, dense
    or sparse, real and finite (require_pair_interaction checks the three
    together), and a missing -d counts as no interaction. name is the
    keyword the caller gave the interaction with (Vr, Jr), for the message.

    n_i n_j and S_i.S_j are symmetric, so a pair has one coupling. What
    reads v does not agree on a v that has two: the Hartree-Fock mean field
    visits a pair from both ends and so averages them, and the
    Bethe-Salpeter kernel, the RPA vertex V(q) and the held pattern of a
    spinon mean field take one direction each, which gives a kernel or a
    Hamiltonian that is not Hermitian. A function symmetric in its two
    arguments fails this when it is not periodic in the unit cell (a
    modulation written with a Cartesian coordinate whose period is not
    the one of the cell), since the pair is then built from two different
    cells. The tolerance is the relative one of the reciprocity check of
    the hoppings in htk/g2h.py."""
    from scipy.sparse import csr_matrix
    v = {tuple(int(x) for x in d): csr_matrix(m) for d,m in v.items()}
    worst,where,scale = 0.,None,0.
    for d,m in v.items():
        if m.nnz>0: scale = max(scale,np.max(np.abs(m.data)))
        back = v.get((-d[0],-d[1],-d[2]),None)
        diff = m.tocoo() if back is None else (m - back.T).tocoo()
        if diff.nnz==0: continue
        k = np.argmax(np.abs(diff.data))
        if np.abs(diff.data[k])>worst:
            worst,where = np.abs(diff.data[k]),(d,diff.row[k],diff.col[k])
    if worst>tol*scale:
        d,i,j = where
        back = v.get((-d[0],-d[1],-d[2]),None)
        forward = _show_coupling(v[d][i,j])
        backward = _show_coupling(0.j if back is None else back[j,i])
        raise ValueError(("%s is not reciprocal: %s = %s %s, and %s = %s "
            "for the same pair from its other end, a mismatch of %.3g; a "
            "pair of sites has one coupling, so %s(r1,r2) must be symmetric "
            "in its two arguments and periodic in the unit cell, "
            "%s(r_i, r_j+R) = %s(r_j, r_i-R) for every lattice vector R "
            "(write a modulation with the fractional coordinates of the "
            "cell)") % (name,name,forward,_pair_text(d,i,j),name,backward,
            worst,name,name,name))


def _interaction_matrices(g,Vr,**kwargs):
    """Vr on every pair of sites that the multicell builder keeps, as a
    spinless {direction: matrix} dictionary. Both directions of each pair
    are evaluated (non_hermitian=True): a Hermitian build takes the cell -d
    as the adjoint of the cell d, so require_pair_interaction would never
    see a Vr that is not reciprocal. The builder is the one of the
    hoppings, and its warning for a function that is not reciprocal gives
    the advice of a hopping (non_hermitian=True), which is wrong for an
    interaction: require_pair_interaction raises for it instead. That
    check of the hoppings also subtracts the values of the two ends, which
    numpy warns about for an infinite one; require_pair_interaction says
    what to do with it, so the warning is not shown."""
    import warnings
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.filterwarnings("ignore",message="the hopping function tij "
                "is not reciprocal")
        hv = g.get_hamiltonian(has_spin=False,is_multicell=True,tij=Vr,
                non_hermitian=True,**kwargs)
    return {tuple(int(x) for x in d): np.array(m,dtype=np.complex128)
            for d,m in hv.get_hopping_dict().items()}


def cells_within(g,rcut):
    """The lattice vectors R = n1 a1 + n2 a2 + n3 a3 of every cell that can
    hold a site closer than rcut to a site of the cell 0, as a list of
    (n1,n2,n3), the cell 0 first. Along each periodic direction the range
    is the extent of the unit cell plus rcut, in units of the spacing of
    the lattice planes of that direction: a pair at a distance d <= rcut
    has n_k = (d + r_i - r_j).b_k, so |n_k| <= rcut |b_k| + max|(r_i-r_j).b_k|.
    Each direction gets its own range, so a long and thin supercell does not
    pay the range of its long side along the short one."""
    import itertools
    nd = g.dimensionality
    if nd==0: return [(0,0,0)]
    r = np.array(g.r)
    A = np.array([g.a1,g.a2,g.a3][:nd]) # periodic lattice vectors
    B = np.linalg.pinv(A) # cell index of a vector R is R@B
    frac = r@B # fractional coordinates of the sites
    extent = np.max(frac,axis=0) - np.min(frac,axis=0)
    # the margin covers the tolerance on the shell distances (1e-4 on the
    # squared distance in distance_hopping_matrix) and the rounding
    ncell = np.floor(extent + rcut*np.linalg.norm(B,axis=0) + 1e-3)
    ranges = [range(-int(n),int(n)+1) for n in ncell] + [range(1)]*(3-nd)
    cells = [tuple(int(x) for x in c) for c in itertools.product(*ranges)]
    cells.remove((0,0,0))
    return [(0,0,0)] + cells


def shell_interaction(g,vs,nd=None):
    """The coupling vs[n] between every pair of sites at the distance of
    the (n+1)-th neighbor shell of g (g.neighbor_distances()), as a
    spinless {direction: matrix} dictionary with m[i,j] the coupling of
    the pair (r_i, r_j + R_d), with every cell that holds such a pair.

    This is the V1/V2/V3 (and J1/J2/J3) shells of the interactions. They
    used to be built with the multicell builder of get_hamiltonian, which
    looks two cells away by default (nc=2), shorter than the third shell
    on a small cell: on the one-site chain it dropped V3 entirely, with no
    warning, in the mean field, the RPA and the Bethe-Salpeter kernel. Here
    the cells are those within the farthest shell with a nonzero coupling
    (cells_within), as for the Vr tail. A direction with no coupling is
    left out, the cell 0 is always there.

    nd: the shell distances, if the caller already has them."""
    if nd is None: nd = g.neighbor_distances()
    nd = np.array(nd,dtype=np.float64)[0:len(vs)]
    vs = np.array(vs)[0:len(nd)]
    used = [d for (d,v) in zip(nd,vs) if v!=0.]
    cells = cells_within(g,max(used)) if len(used)>0 else [(0,0,0)]
    mgenerator = distance_hopping_matrix(vs,nd)
    r = np.array(g.r,dtype=np.float64)
    out = dict()
    for d in cells:
        R = d[0]*np.array(g.a1) + d[1]*np.array(g.a2) + d[2]*np.array(g.a3)
        m = np.array(mgenerator(r,r+R),dtype=np.complex128)
        if d==(0,0,0) or np.any(m!=0.): out[d] = m
    return out


def distance_cut_interaction(g,Vr,rcut=None,name="Vr",onsite=True):
    """The pair interaction Vr(r1,r2) between every pair of sites closer
    than rcut (a pair at exactly rcut is kept), as a spinless
    {direction: matrix} dictionary with m[i,j] = Vr(r_i, r_j + R_d)

    A pair has one coupling, a real and finite number, so Vr must give the
    same one from the two ends of a pair, Vr(r_i, r_j+R) = Vr(r_j, r_i-R);
    otherwise this raises (require_pair_interaction), with name the keyword
    of the caller in the message. onsite=False drops the pair of a site
    with itself before that check, for a caller that does not use it (the
    Bethe-Salpeter kernel, since an electron does not interact with
    itself), so that a tail that diverges at zero distance is taken there.

    rcut=None keeps every pair of a finite (0d) geometry, whatever its
    distance, and means rcut=5.0 for a periodic one, the range the
    multicell builder uses by default. An explicit rcut applies in both.

    The multicell builder behind get_hamiltonian(tij=...) keeps a whole
    lattice cell as soon as one of its pairs is within its rcut, so on its
    own it returns every pair below rcut plus an uneven fringe beyond it:
    on the honeycomb lattice it kept 4 of the 6 pairs at distance 5 and
    4 of the 12 at 5.29, which breaks the three-fold rotation of the
    lattice. Here every cell that can hold a pair within rcut is built,
    and then every pair beyond rcut is dropped, so each distance shell is
    kept or dropped whole."""
    def checked(out): # what every exit returns
        if not onsite: np.fill_diagonal(out[(0,0,0)],0.)
        require_pair_interaction(out,name=name)
        return out
    nd = g.dimensionality
    if rcut is None:
        if nd==0: # a finite system has no fringe, every pair is kept
            return checked(_interaction_matrices(g,Vr))
        rcut = 5.0
    rcut = float(rcut)
    if not rcut>0.:
        raise ValueError("rcut, the range of the Vr interaction, must be "
          +"positive, got "+str(rcut))
    tol = 1e-6 # a shell sitting exactly at rcut is kept whole
    r = np.array(g.r)
    if nd>0: # number of cells needed to reach every pair within rcut
        A = np.array([g.a1,g.a2,g.a3][:nd]) # periodic lattice vectors
        B = np.linalg.pinv(A) # cell index of a vector R is R@B
        span = np.max(np.linalg.norm(r[:,None,:]-r[None,:,:],axis=2))
        ncells = int(np.ceil((rcut+span)*np.max(np.linalg.norm(B,axis=0))))+1
    else: ncells = 0
    out = dict()
    for d,m in _interaction_matrices(g,Vr,cutoff=ncells,
            rcut=rcut+10*tol).items():
        R = d[0]*g.a1 + d[1]*g.a2 + d[2]*g.a3
        dist = np.linalg.norm(r[None,:,:]+R-r[:,None,:],axis=2) # |r_j+R-r_i|
        m[dist>rcut+tol] = 0.
        # != and not a maximum: a NaN must stay, for the check to see it
        if np.any(m!=0.) or d==(0,0,0): out[d] = m
    return checked(out)


def folded_interaction(g,Jij=None,rcut=None,name="Jij"):
    """The couplings of a classical model (spins, Ising, lattice gas)
    between the pairs of sites of the cell g, with the periodic images
    folded onto the cell, as the k=0 matrix in coo form.

    A function Jij(r1,r2) is taken on every pair within rcut, each distance
    shell whole, as the Vr of the mean field (distance_cut_interaction:
    rcut=None is every pair of a finite system and 5.0 for a periodic one).
    It used to go through the single-cell storage of g.get_hamiltonian,
    which in 1D and 2D keeps only the neighboring cells, so a coupling two
    cells away on a small cell was dropped, and on a large cell the pairs
    kept depended on where the site sits in the cell. A HoppingGenerator
    (NNG and the others) is evaluated the same way on every cell within
    rcut (generator_within). A list of shell couplings or None goes through
    g.get_hamiltonian(tij=Jij) as before, with the range of its shells;
    rcut is for a function or a generator only"""
    from scipy.sparse import coo_matrix
    if isinstance(Jij,HoppingGenerator): # a matrix of couplings per cell
        return coo_matrix(generator_within(g,Jij.f,rcut=rcut))
    if callable(Jij):
        ms = distance_cut_interaction(g,Jij,rcut=rcut,name=name)
        return coo_matrix(sum(ms.values()))
    if rcut is not None:
        raise ValueError("rcut is the range of a coupling given as a "
                "function Jij(r1,r2), and "+name+"="+repr(Jij)+" is not "
                "one; a list of shell couplings has the range of its shells")
    h = g.get_hamiltonian(has_spin=False,tij=Jij)
    return coo_matrix(h.get_hk_gen()([0.,0.,0.])) # the k=0 matrix


def generator_within(g,fm,rcut=None):
    """The couplings of a matrix generator fm(rs1,rs2), the coupling of
    every position of rs1 with every position of rs2, between the sites of
    g and those of every cell that holds a pair within rcut, each pair
    beyond rcut dropped, folded onto the cell (the k=0 matrix). rcut=None
    is every pair of a finite system and 5.0 for a periodic one, as in
    distance_cut_interaction"""
    r = np.array(g.r)
    if g.dimensionality==0 and rcut is None: return np.array(fm(r,r))
    rcut = 5.0 if rcut is None else float(rcut)
    out = 0.
    for d in cells_within(g,rcut):
        R = d[0]*np.array(g.a1) + d[1]*np.array(g.a2) + d[2]*np.array(g.a3)
        m = np.array(fm(r,r+R),dtype=np.complex128)
        dist = np.linalg.norm(r[None,:,:]+R-r[:,None,:],axis=2) # |r_j+R-r_i|
        m[dist>rcut+1e-6] = 0. # a shell sitting at rcut is kept whole
        out = out + m
    return out


def add_distance_cut_interaction(v,g,Vr,rcut=None,name="Vr",onsite=True):
    """Add the Vr interaction of distance_cut_interaction to the spinless
    {direction: matrix} dictionary v, in place, and return v"""
    for d,m in distance_cut_interaction(g,Vr,rcut=rcut,name=name,
            onsite=onsite).items():
        v[d] = v[d] + m if d in v else m
    return v


@jit(nopython=True,cache=True)
def distance_hopping_matrix_jit(r1,r2,vs,ds2,out):
    """Return a hopping that to the 1-th neighbor is vs"""
    n = len(r1) # number of sites
    nn = len(ds2) # number of neighbors
    for i in range(n):
      for j in range(n):
          dr = r1[i] - r2[j] # difference
          dr2 = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]
          for k in range(nn):
              if np.abs(ds2[k]-dr2)<1e-4: out[i,j] = vs[k]
    return out



def strained_hopping(g,t=1.0,dt=0.0,f=None,**kwargs):
    """Return first neighbor hoppings with strain"""
    if f is None: # no function provided
        from . import potentials
        f = potentials.commensurate_potential(g,average=t,amplitude=dt,
                **kwargs)
    def fout(r1,r2):
        dr = r1-r2
        dr2 = dr.dot(dr)
        if .9<dr2<1.1: return f((r1+r2)/2.)
        else: return 0.0
    return fout


def strained_hopping_matrix(*args,**kwargs):
    f = strained_hopping(*args,**kwargs)
    return entry2matrix(f) # return the matrix


from .algebra import isnumber

class HoppingGenerator():
    """Class for a Hopping generator"""
    def __init__(self,m):
        if type(m)==HoppingGenerator: self.f = m.f # redefine
        elif callable(m): self.f = m # define callable function
        else:
            raise TypeError("a HoppingGenerator must be built from another "
                    "HoppingGenerator or a callable")
    def __call__(self,rs1,rs2):
        """Call method"""
        return self.f(rs1,rs2)
    def __mul__(self,m):
        if isnumber(m): # number input 
            fout = lambda rs1,rs2: np.array(m*np.array(self(rs1,rs2)))
        else: # anything else
            m2 = HoppingGenerator(m) # transform to HoppingGenerator
            fout = lambda rs1,rs2: np.array(self(rs1,rs2))*np.array(m2(rs1,rs2))
        return HoppingGenerator(fout) # return new object
    def __rmul__(self,m): return self*m # commutative 
    def __radd__(self,m): return self + m # commutative 
    def __add__(self,m):  
        m2 = HoppingGenerator(m) # transform to HoppingGenerator
        fout = lambda rs1,rs2: np.array(self(rs1,rs2)) + np.array(m2(rs1,rs2))
        return HoppingGenerator(fout) # return new object
    def __neg__(self): return (-1)*self # minus
    def __sub__(self,a): return self + (-1)*a # minus
    def copy(self):
        from copy import deepcopy
        return deepcopy(self)
    def apply(self,f):
        """Given a certain function of the position, 
        apply it to all the hoppings"""
        def fm(rs1,rs2): # define a new function
            m = self.f(rs1,rs2) # get the matrix
            p = np.array([[f((r1+r2)/2.) for r1 in rs1] for r2 in rs2])
            return m*p # return product
        return HoppingGenerator(fm) # new generator




def twisted_matrix_python(cutoff=10,**kwargs):
  """Function returning the hopping of a twisted matrix"""
  def tij(rs1,rs2): # function to return
      nr = len(rs1) # length
      nmax = nr*int(10*cutoff**2) # maximum number of hoppings
      data = np.zeros(nmax,dtype=np.complex128) # data
      ii = np.zeros(nmax,dtype=int) # index
      jj = np.zeros(nmax,dtype=int) # index
      ii,jj,data,nk = twisted_matrix_jit(np.array(rs1),np.array(rs2),
                         ii,jj,data,cutoff=cutoff,**kwargs) # call function
      if nk>nmax: # sanity check
          raise MemoryError("more hoppings were found than the preallocated "
                  "arrays hold, increase the estimate in tij")
      ii = ii[0:nk] # only nonzero
      jj = jj[0:nk] # only nonzero
      data = data[0:nk] # only nonzero
      out = csc_matrix((data,(ii,jj)),shape=(nr,nr),dtype=np.complex128) # matrix
      out.eliminate_zeros()
      return out
  return tij
    


@jit(nopython=True,cache=True)
def twisted_matrix_jit(rs1,rs2,ii,jj,data,cutoff=5.0,ti=0.3,lambi=8.0,
        mint = 1e-5,t=1.0,
        lamb=12.0,dl=3.0,lambz=10.0):
  """Hopping for twisted bilayer graphene, returning the indexes of a 
  sparse matrix"""
  cutoff2 = cutoff**2 # cutoff in distance
  ik = 0 # counter
  for i1 in range(len(rs1)):
    for i2 in range(len(rs2)):
      r1 = rs1[i1]
      r2 = rs2[i2]
      dr = r1-r2 # distance
      rr = dr[0]**2 + dr[1]**2 + dr[2]**2 # distance
      if rr>cutoff2: continue  # too far
      if rr<0.001: continue # same atom
      dx = r1[0]-r2[0]
      dy = r1[1]-r2[1]
      dz = r1[2]-r2[2]
      r = np.sqrt(rr)
  #    if r2>100.0: return 0.0 # too far
      if (r-1.0)<-0.1:
          raise ValueError("two sites are closer than the first-neighbor "
                  "distance this hopping is normalized to")
      out = -t*(dx*dx + dy*dy)/rr*np.exp(-lamb*(r-1.0))*np.exp(-lambz*dz*dz)
      out += -ti*(dz*dz)/rr*np.exp(-lambi*(r-dl))
      #### fix for magnetic field
#      cphi = np.cos(phi*np.pi)
#      sphi = np.sin(phi*np.pi)
#      r = (r1+r2)/2.
#      dr = r1-r2
#      p = 2*r[2]*(dr[0]*sphi - dr[1]*cphi)
#      out *= np.exp(1j*b*p)
      #####
      if np.abs(out)>mint:
        ii[ik] = i1 # store
        jj[ik] = i2 # store
        data[ik] = out # store
        ik += 1 # increase counter
  return ii,jj,data,ik





def NNG(g,ts):
    """Hopping generator for nearest neighbor hoppings
    - g: geometry
    - ts: hoppings for each neighbor
    """
    fm = neighbor_hopping_matrix(g,ts) # get the function
    return HoppingGenerator(fm) # return a generator 

def ILG(g,ti,**kwargs):
    """Generator for an interlayer hopping using an exponential
    parametrization
    - g: geometry
    - ti: interlayer hopping
    - **kwargs: optional arguments for twisted hopping
    """
    from .potentials import Potential
    from . import algebra
    if callable(ti): ti = Potential(ti) # transform to potential
    elif algebra.isnumber(ti): pass
    else:
        raise TypeError("the interlayer hopping amplitude must be a number or "
                "a callable of the position")
    fm = twisted_matrix(t=0.,ti=-1*ti,**kwargs) # interlayer hopping generator
    # return a generator
    return HoppingGenerator(lambda *args: algebra.todense(fm(*args))) 
