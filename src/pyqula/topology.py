# library to calculate topological properties
import numpy as np
from scipy.sparse import bmat, csc_matrix
import scipy.linalg as lg
import scipy.sparse.linalg as slg
from . import multicell
from . import klist
from . import operators
from . import inout
from . import timing
from . import algebra
from . import parallel
from numba import jit

arpack_tol = algebra.arpack_tol
arpack_maxiter = algebra.arpack_maxiter





def get_berry_curvature_path(h,kpath=None,dk=0.01,
      window=None,max_waves=None,nk=600,
      mode="Wilson",delta=0.001,reciprocal=False,operator=None,
      silent = True):
    """Calculate and write in file the Berry curvature
    in a certain kpath"""
    operator = get_operator(h,operator)
    kpath = klist.get_kpath(h.geometry,kpath=kpath,nk=nk) # take default kpath
    tr = timing.Testimator("BERRY CURVATURE",silent=silent)
    ik = 0
    if operator is not None: mode="Green" # Green function mode
    if mode=="Green": # build the generator once, not per k-point
        f = h.get_gk_gen(delta=delta) # get generator
        def gI(e=0.,k=[0.,0.,0.]): return f(e=e,k=k,inv=True) # exact inverse
    def getb(k):
      if reciprocal:  k = h.geometry.get_k2K_generator()(k) # convert
      if mode=="Wilson":
        b = berry_curvature(h,k,dk=dk,window=window,max_waves=max_waves)
      elif mode=="Green":
        b = berry_green(f,k=k,operator=operator,gI=gI)
      else:
        raise ValueError("unknown mode for the Berry curvature; the accepted "
                "ones are 'Wilson' and 'Green'")
      return str(k[0])+"   "+str(k[1])+"   "+str(b)+"\n"
    fo = open("BERRY_CURVATURE.OUT","w") # open file
    if parallel.cores==1: # serial execution
      for k in kpath:
        tr.remaining(ik,len(kpath))
        ik += 1
        fo.write(getb(k)) # write result
        fo.flush()
    else: # parallel execution
        out = parallel.pcall(getb,kpath)
        for o in out: fo.write(o) # write
    fo.close() # close file
    m = np.genfromtxt("BERRY_CURVATURE.OUT").transpose()
    return np.array(range(len(m[0]))),m[2]


# alias for compatibility
write_berry = get_berry_curvature_path






def berry_phase(h,nk=20,kpath=None,write=True):
    """ Calculates the Berry phase of a Hamiltonian

    SIGN CONVENTION. This returns +gamma in the standard discrete
    convention gamma = -Im log prod_j <u_j|u_{j+1}> (King-Smith &
    Vanderbilt, PRB 47, 1651(R) (1993)), i.e. the Berry phase of the
    connection A = i<u|grad_k u>. The negation that formula carries is
    supplied by uij, not by the arctan2 below -- see berry_curvature's
    SIGN CONVENTION docstring for the derivation. Checked against a direct
    King-Smith-Vanderbilt evaluation on closed k-loops (six digits).
    """
    if h.dimensionality==0:
        raise ValueError("a 0-dimensional Hamiltonian has no Brillouin "
          +"zone, so it has no Berry phase")
    elif h.dimensionality == 1:
      ks = np.linspace(0.,1.,nk,endpoint=False) # list of kpoints
      ks = np.array([[k,0.,0.] for k in ks]) # redefine
    elif h.dimensionality > 1: # you must provide a kpath
        if kpath is None:
            raise ValueError("in more than one dimension the Berry phase "
              +"depends on the path, so you must provide kpath=")
        ks = kpath # continue
        nk = len(kpath) # redefine
    else: # otherwise
      raise ValueError("the Berry phase needs a Hamiltonian with a "
              "non-negative dimensionality")
    hkgen = h.get_hk_gen() # get Hamiltonian generator
    wf0 = occupied_states(hkgen,ks[0]) # get occupied states, first k-point
    if len(wf0)==0:
        raise ValueError("there is no occupied state (no eigenvalue below "
          +"zero) at the first k-point, so the Berry phase of the occupied "
          +"manifold is not defined; shift the Fermi energy into the gap "
          +"with h.shift_fermi / h.set_filling first")
    wfold = wf0.copy() # copy
    m = np.array(np.identity(len(wf0))) # initialize as the identity matrix
    for ik in range(1,len(ks)): # loop over k-points, except first one
      wf = occupied_states(hkgen,ks[ik])  # get waves
      if len(wf)!=len(wf0):
        # the manifold must be the same size all along the path, otherwise
        # the link variables are not square and this used to die with an
        # opaque matmul shape error
        raise ValueError("the number of occupied states changes along the "
          +"k-path ("+str(len(wf0))+" at the first k-point, "+str(len(wf))
          +" at k="+str(ks[ik])+"), so the occupied manifold is not "
          +"separated by a gap and its Berry phase is not defined. Shift "
          +"the Fermi energy into a gap, or pass an energy window")
      m = m@uij(wfold,wf)   # get the uij   and multiply
      wfold = wf.copy() # this is the new old
    m = m@uij(wfold,wf0)   # last one
    d = lg.det(m) # calculate determinant
    phi = np.arctan2(d.imag,d.real)
    if write: open("BERRY_PHASE.OUT","w").write(str(phi/np.pi)+"\n")
    return phi # return Berry phase








def _chiral_operator(h,chiral):
    """Dense matrix of the chiral operator S of a one-dimensional
    Hamiltonian, checked to be one: S^2=1 and S H(k) S = -H(k)

    chiral=None takes sigma_y tau_y on every site of a Nambu Hamiltonian,
    the chiral symmetry of a BdG Hamiltonian that is real in real space
    (class BDI, a Kitaev chain), and the sublattice operator otherwise"""
    n = h.intra.shape[0]
    if chiral is None:
        if h.has_eh:
            sy = np.array([[0.,-1j],[1j,0.]])
            S = np.kron(np.identity(n//4),np.kron(sy,sy)) # tau_y sigma_y
        elif h.geometry.has_sublattice:
            S = h.get_operator("sublattice").get_matrix()
        else:
            raise ValueError("the winding number needs a chiral operator, "
              +"and this Hamiltonian is not a Nambu one and has no "
              +"sublattice to take it from; pass chiral=, a name, a "
              +"matrix or an Operator, or label the sublattice with "
              +"g.get_sublattice()")
    else: S = get_operator(h,chiral).get_matrix()
    S = np.array(algebra.todense(S),dtype=complex)
    if np.max(np.abs(S@S-np.identity(n)))>1e-6:
        raise ValueError("the chiral operator has to square to the identity")
    hkgen = h.get_hk_gen()
    for k in [0.,0.5,0.1234,0.3791]: # S has to anticommute with H(k)
        hk = np.array(algebra.todense(hkgen([k,0.,0.])),dtype=complex)
        r = np.max(np.abs(S@hk+hk@S))
        if r>1e-6*max(1.,np.max(np.abs(hk))):
            raise ValueError("the chiral operator does not anticommute with "
              +"the Hamiltonian, the residual of S H(k) + H(k) S is "+str(r)
              +" at k="+str(k)+", so this Hamiltonian has no such chiral "
              +"symmetry and no winding number")
    return S


def winding_number(h,chiral=None,nk=200):
    """Winding number of a one-dimensional Hamiltonian with a chiral
    symmetry (classes AIII and BDI)

    In the eigenbasis of the chiral operator S, H(k) is off-diagonal, with
    a block q(k) = P_+^dagger H(k) P_- between the states of S=+1 and S=-1,
    and the winding number is that of det q(k) around the Brillouin zone,
    W = (1/2 pi i) int dk d log det q(k) (Ryu-Schnyder-Furusaki-Ludwig,
    arXiv:0912.2157). It counts the zero modes at each end of an open
    chain, on one sublattice, an integer where the Zak phase gives only its
    parity. No eigenvector of H enters, so there is no gauge to fix.

    chiral: None (see _chiral_operator), or a name, a matrix or an Operator
    nk: k-points around the zone; each step of the phase of det q must stay
        below pi

    The sign of W follows the sign of S, and a gap closing, a k-point
    where det q vanishes, raises."""
    if h.dimensionality!=1:
        raise ValueError("the winding number needs a one-dimensional "
          +"Hamiltonian, and this one has dimensionality "
          +str(h.dimensionality))
    S = _chiral_operator(h,chiral)
    (s,v) = np.linalg.eigh(S)
    pp,pm = v[:,s>0],v[:,s<0] # the S=+1 and S=-1 subspaces
    if pp.shape[1]!=pm.shape[1]:
        raise ValueError("the chiral operator has "+str(pp.shape[1])
          +" eigenvalues +1 and "+str(pm.shape[1])+" eigenvalues -1, and "
          +"the winding number needs as many of each")
    hkgen = h.get_hk_gen()
    ks = np.linspace(0.,1.,nk,endpoint=False)
    dets,smin = [],[]
    for k in ks:
        q = pp.conj().T@np.array(algebra.todense(hkgen([k,0.,0.])))@pm
        dets.append(np.linalg.det(q))
        smin.append(np.min(np.linalg.svd(q,compute_uv=False)))
    if np.min(smin)<1e-8*max(1.,np.max(smin)):
        raise ValueError("the gap closes at k="+str(ks[np.argmin(smin)])
          +", so the winding number is not defined")
    phi = np.angle(dets)
    dphi = np.diff(np.concatenate([phi,phi[:1]])) # closed loop
    dphi = (dphi+np.pi)%(2.*np.pi) - np.pi
    return int(np.round(np.sum(dphi)/(2.*np.pi)))


def z2_invariant_1d(h,nk=200):
    """Z2 invariant of a one-dimensional time-reversal-symmetric
    superconductor (class DIII): +1 trivial, -1 with a Kramers pair of
    Majorana zero modes at each end

    It is the Kramers polarization of Budich-Ardonne (arXiv:1308.1256,
    built on cond-mat/0606336),

        nu = det(U) Pf theta(0) / Pf theta(1/2),

    where U is the product of the overlaps of the negative-energy states
    from k=0 to k=1/2 (the Kato propagator, each link replaced by its
    unitary part so that nu converges faster in nk) and theta is the time
    reversal T = i sigma_y K restricted to those states at the two
    time-reversal-invariant momenta, where it is antisymmetric. Any basis
    of the negative-energy states is allowed at every k-point, so there is
    no gauge to fix. A global phase of the pairing is removed first, as in
    has_time_reversal_symmetry, since T is exact only without it.

    nk: k-points from 0 to 1/2

    nu is +-1 up to the discretization of the product, and a value far
    from both raises, asking for a larger nk."""
    from .htk.symmetry import _remove_pairing_phase
    from .topologytk.pfaffian import pfaffian
    if h.dimensionality!=1:
        raise ValueError("this Z2 invariant needs a one-dimensional "
          +"Hamiltonian, and this one has dimensionality "
          +str(h.dimensionality))
    from .check import require_nambu
    require_nambu(h,"the Z2 invariant of a class DIII superconductor")
    if not h.has_time_reversal_symmetry():
        raise ValueError("the Z2 invariant of a class DIII superconductor "
          +"needs a time-reversal-symmetric Hamiltonian, and this one "
          +"breaks time reversal")
    h1 = _remove_pairing_phase(h) # T = i sigma_y K is exact on this copy
    n = h1.intra.shape[0]
    UT = np.kron(np.identity(n//2),np.array([[0.,1.],[-1.,0.]])) # i sigma_y
    hkgen = h1.get_hk_gen()
    def occ(k): # negative-energy states, as columns
        (e,v) = np.linalg.eigh(np.array(algebra.todense(hkgen([k,0.,0.]))))
        return v[:,e<0.]
    ks = np.linspace(0.,0.5,nk+1) # from k=0 to k=1/2, both included
    vs = [occ(k) for k in ks]
    U = np.identity(vs[0].shape[1],dtype=complex)
    for j in range(nk): # ordered from right to left with increasing k
        (a,sv,b) = np.linalg.svd(vs[j+1].conj().T@vs[j])
        U = (a@b)@U # unitary part of the link
    def theta(v): # time reversal on the states v, antisymmetric at a TRIM
        t = v.conj().T@UT@v.conj()
        return (t-t.T)/2.
    nu = np.linalg.det(U)*pfaffian(theta(vs[0]))/pfaffian(theta(vs[-1]))
    if abs(nu-np.sign(nu.real))>0.2:
        raise ValueError("the Z2 invariant came out as "+str(nu)+", not "
          +"close to +1 or -1, so the product over the k-points is not "
          +"converged or the gap closes; raise nk")
    return int(np.sign(nu.real))


def berry_curvature(h,k,dk=0.01,window=None,max_waves=None):
  """ Calculates the Berry curvature of a 2d hamiltonian

  SIGN CONVENTION. This returns +Omega in the convention A = i<u|grad_k u>,
  Omega = curl A of Xiao, Chang & Niu, RMP 82, 1959 (2010) -- the same sign
  as the textbook Kubo formula

      Omega_n = -2 Im sum_m <n|dH/dkx|m><m|dH/dky|n>/(E_n-E_m)^2

  -- and every Chern number built on it (precise_chern, mesh_chern,
  chern_qtci, berry_map, ...) inherits that sign. Note that this is the
  argument of the closed link-variable product taken directly, with no
  explicit minus sign anywhere below, even though the standard discrete
  Berry phase is gamma = -Im log prod_j <u_j|u_{j+1}> (King-Smith &
  Vanderbilt, PRB 47, 1651(R) (1993)). The negation is already in uij:
  occstates.occupied_states hands back ALREADY-CONJUGATED wavefunctions and
  overlap.uij conjugates its first argument a second time, so
  uij(a,b)[i,j] = <b_j|a_i> = conj(<a_i|b_j>). The closed product built
  from it is therefore the complex conjugate of prod_j <u_j|u_{j+1}> =
  exp(-i*gamma), and its argument is +gamma.

  Measured, not assumed: the ratio of this function to an independent
  finite-difference Kubo evaluation of the formula above is 1.000000 at
  every k tested on a gapped Haldane model, the Kubo reference itself
  having been calibrated on the Provost-Vallee spin-1/2 example (upper band
  integrating to -2*pi over the sphere), and berry_phase reproduces the
  King-Smith-Vanderbilt gamma to six digits on closed k-loops. pyqula's
  Bloch convention is H(k) = sum_R t(R) exp(2*pi*i*k.R) with k in reduced
  coordinates, so the returned curvature is per unit area of the reduced
  k-plane; for a right-handed lattice basis (the usual case) that carries
  the same sign as the Cartesian one.

  A code that adopts the opposite convention, A = -i<u|grad_k u>, reports
  the opposite curvature and Chern signs; check the convention before
  comparing pyqula's output with another package's.
  """
  if h.dimensionality != 2: # only for 2d
    raise ValueError("the Berry curvature is only defined for 2d Hamiltonians")
  k = np.array([k[0],k[1]]) 
  dx = np.array([dk,0.])
  dy = np.array([0.,dk])
# get the function that returns the occ states
  occf = occ_states_generator(h,k,window=window,max_waves=max_waves)  
  # get the waves
#  print("Doing k-point",k)
  wf1 = occf(k-dx-dy) 
  wf2 = occf(k+dx-dy) 
  wf3 = occf(k+dx+dy) 
  wf4 = occf(k-dx+dy) 
  dims = [len(wf1),len(wf2),len(wf3),len(wf4)] # number of vectors
  if max(dims)!=min(dims): # check that the dimensions are fine 
#    print("WARNING, skipping this k-point",k)
    return 0.0 # if different number of vectors
  if max(dims)==0:
    # no occupied state at this k-point (e.g. every band above the Fermi
    # energy): the Berry curvature is zero, not an error -- this used to
    # reach uij with empty arrays and die with an opaque TypeError
    return 0.0
  # get the uij  
  m = uij(wf1,wf2)@uij(wf2,wf3)@uij(wf3,wf4)@uij(wf4,wf1)
  d = lg.det(m) # calculate determinant
  phi = np.arctan2(d.imag,d.real)/(4.*dk*dk)
  return phi


from .topologytk.occstates import occ_states_generator
from .topologytk.occstates import occupied_states
from .topologytk.occstates import occ_states2d


from .topologytk.overlap import uij


def precise_chern(h,dk=0.01, mode="Wilson",delta=0.0001,operator=None,
        nk=None):
    """ Calculates the chern number of a 2d system """
    from scipy import integrate
    if nk is not None: # every sibling Chern path takes one; this one cannot
        raise ValueError("precise_chern integrates the Brillouin zone "
            "adaptively (scipy.integrate.dblquad), so it has no k-mesh and "
            "nk is meaningless here; got nk="+str(nk)+". Use dk to set the "
            "finite-difference step, or h.get_chern(nk=...) for the "
            "fixed-mesh Chern number")
    operator = get_operator(h,operator) # accept a name, matrix or callable
    if operator is not None and mode=="Wilson":
        # the Wilson branch below calls berry_curvature without the
        # operator, so an operator-projected Chern number asked for in this
        # mode used to come back as the unprojected one
        raise ValueError("precise_chern only honours operator= in "
            "mode='Green'; got mode='Wilson', whose Wilson-loop curvature "
            "has no projected form here. Pass mode='Green' (slower, it "
            "integrates a Green's function adaptively) or use "
            "h.get_chern(operator=...), which switches for you")
    err = {"epsabs" : 1.0, "epsrel": 1.0,"limit" : 10}
    if mode=="Green": # build the generator once, not per (x,y) evaluation
        f2 = h.get_gk_gen(delta=delta) # get generator
        def gI2(e=0.,k=[0.,0.,0.]): return f2(e=e,k=k,inv=True) # exact inverse
    def f(x,y): # function to integrate
      if mode=="Wilson":
        return berry_curvature(h,np.array([x,y]),dk=dk)
      elif mode=="Green":
         return berry_green(f2,k=[x,y,0.],operator=operator,gI=gI2)
      else:
        raise ValueError("unknown mode for the Berry curvature; the accepted "
                "ones are 'Wilson' and 'Green'")
    c = integrate.dblquad(f,0.,1.,lambda x : 0., lambda x: 1.,epsabs=0.01,
                            epsrel=0.01)
    chern = c[0]/(2.*np.pi)
    open("CHERN.OUT","w").write(str(chern)+"\n")
    return chern


def mesh_chern(h,dk=-1,nk=10,delta=0.0001,mode="Wilson",
        operator=None,kmesh=None):
    """ Calculates the chern number of a 2d system """
    c = 0.0
    ks = [] # array for kpoints
    bs = [] # array for berrys
    operator = get_operator(h,operator) # accept a name, matrix or callable
    if dk<0: dk = 1./float(2*nk) # automatic dk
    if kmesh is not None: # infer the dk of the mesh
        dk = klist.infer_kmesh_dk(kmesh,d=2)
    if operator is not None and mode=="Wilson":
      print("Switching to Green mode in topology")
      mode="Green"
    # create the function
    if mode=="Green": # build the generator once, not per k-point
        f2 = h.get_gk_gen(delta=delta) # get generator
        def gI2(e=0.,k=[0.,0.,0.]): return f2(e=e,k=k,inv=True) # exact inverse
    def fberry(k): # function to integrate
      if mode=="Wilson":
        return berry_curvature(h,k,dk=dk)
      if mode=="Green":
         return berry_green(f2,k=[k[0],k[1],0.],operator=operator,gI=gI2)
    ##################
    if kmesh is None: # no kmesh provided
        ks = klist.kmesh(h.dimensionality,nk=nk) # get the mesh
    else: ks = kmesh # use the provided kmesh
    ik = 0
    from .topologytk.berry import use_berry_curvature_mesh
    if use_berry_curvature_mesh(h,mode=mode): # batched, numba-parallel path
        from .topologytk.berry import berry_curvature_mesh
        bs = berry_curvature_mesh(h,ks,dk=dk)
    else: # per-kpoint dispatch
        bs = parallel.pcall(fberry,ks) # compute all the Berry curvatures
    # write in file
    fo = open("BERRY_CURVATURE.OUT","w") # open file
    for (k,b) in zip(ks,bs):
      fo.write(str(k[0])+"   ")
      fo.write(str(k[1])+"   ")
      fo.write(str(b)+"\n")
    fo.close() # close file
    ################
    c = np.sum(bs) # sum berry curvatures
    if kmesh is None: # no kmesh provided
        c = c/(2.*np.pi*nk*nk) # normalize
    else: # kmesh is given
        den = klist.infer_kmesh_density(kmesh,d=2) # infer the volume
        c = den*c/(2.*np.pi) # normalize
    open("CHERN.OUT","w").write(str(c)+"\n")
    return c


def chern_qtci(h,mode="Wilson",delta=0.0001,dk=-1,operator=None,
        nk=20,tolerance=1e-6,**kwargs):
    """Compute the Chern number of a 2D system by integrating the Berry
    curvature over the BZ with qutecipy, instead of summing it over a
    k-point mesh (see mesh_chern). qutecipy approximates the integrand as
    a low-rank tensor train (tensor cross interpolation) and folds a
    Gauss-Kronrod quadrature rule into it.

    ACCURACY -- READ BEFORE CHOOSING THIS OVER mesh_chern. This path is
    accurate for a SMOOTH Berry curvature and unreliable for a sharply
    peaked one, which is the opposite of what an "adaptive" method might be
    expected to give. Measured on a spinful Haldane model (exact C=2),
    error in the returned Chern number:

        smooth   (t2=0.3):  nk=10 4.8e-5   nk=20 5.1e-5   nk=40 9.1e-6
        trivial  (C=0):     nk=10 4.8e-7   nk=20 3.1e-8   nk=40 8.4e-9
        sharp    (t2=0.05): nk=10 6.5e-3   nk=20 1.5e-2   nk=40 6.6e-2

    In the sharp/small-gap case the error GROWS with nk, and raising
    `tolerance` does not help (flat from 1e-4 to 1e-8); mode="Green",
    which has no plaquette at all, degrades the same way, so this is a
    property of the tensor-cross-interpolation quadrature and not of the
    dk below. The likely mechanism is that refining the Gauss-Kronrod grid
    shrinks the fraction of nodes near the curvature peak, so the
    cross-interpolation pivot search can miss it while its rank tolerance
    is still satisfied on the smooth bulk.

    Prefer mesh_chern (integration="grid", the default) whenever the gap is
    small: it is not merely more accurate there but EXACTLY quantized, since
    the Fukui-Hatsugai-Suzuki construction counts vortices in link variables
    rather than quadraturing a field -- measured error ~1e-15 at every nk.
    Use this path for smooth curvature, where it reaches 1e-5..1e-9 from far
    fewer evaluations than a dense mesh.

    nk sets the resolution of the underlying quadrature (mirroring the nk
    k-point-mesh density used elsewhere in this module), via a
    Gauss-Kronrod order growing logarithmically with it:
    GKorder=4*bits+1 with bits=ceil(log2(nk)). If dk is not given
    explicitly it also sets the Wilson-loop plaquette size dk=1/(2*nk).

    See tests/topology/test_chern_qtci_accuracy.py, which pins the smooth
    case and records the sharp-case limitation."""
    from .qtcitk.gkintegrate import gkorder_from_nk, integrate_robust
    operator = get_operator(h,operator) # accept a name, matrix or callable
    if dk<0: dk = 1./float(2*nk) # automatic dk, tied to the quadrature resolution
    GKorder = gkorder_from_nk(nk)
    if operator is not None and mode=="Wilson":
        mode = "Green" # operator-resolved curvature needs Green's function mode
    if mode=="Green": # Green's function generator, built once
        fgk = h.get_gk_gen(delta=delta)
        def gI(e=0.,k=[0.,0.,0.]): return fgk(e=e,k=k,inv=True) # exact inverse
    def f(k):
        if mode=="Wilson": return berry_curvature(h,k,dk=dk)
        elif mode=="Green": return berry_green(fgk,k=[k[0],k[1],0.],operator=operator,gI=gI)
        else:
          raise ValueError("unknown mode for the Berry curvature; the "
                  "accepted ones are 'Wilson' and 'Green'")
    c = integrate_robust(np.float64,f,GKorder,tolerance,**kwargs)
    c = c/(2.*np.pi) # normalize so that the integral gives an integer
    open("CHERN.OUT","w").write(str(c)+"\n")
    return c


def get_berry_curvature(self,**kwargs):
    """Compute Berry curvature"""
    if self.non_hermitian: # non Hermitian case
        from .nonhermitiantk.nhmethods import get_berry_curvature as BNH
        return BNH(self,**kwargs)
    else: # Hermitian case
        return get_berry_curvature_master(self,**kwargs)





def get_berry_curvature_master(h,dk=None,nk=100,
        reciprocal=True,nsuper=1,window=None,
        kpath=None,
               max_waves=None,mode="Wilson",delta=0.001,operator=None,
               write=True,verbose=0):
    """ Return the Berry curvature in 2D reciprocal space """
    operator = get_operator(h,operator) # accept a name, matrix or callable
    # get the right kpoints
    if kpath is None: # no kpath, just to a grid
        ks = [] # list with kpoints
        for x in np.linspace(-nsuper,nsuper,nk,endpoint=False):
          for y in np.linspace(-nsuper,nsuper,nk,endpoint=False):
              ks.append([x,y,0.])
    else: # kpath provided
        ks = h.geometry.get_kpath(kpath,nk=nk)
        reciprocal = False # if given, assume they are in standard way
    ks = np.array(ks) # convert to array
    ##############
    if operator is not None: mode="Green" # Green function mode
    c = 0.0
    if dk is None: dk = 1./float(2*nk) # automatic dk
    if reciprocal: R = np.array(h.geometry.get_k2K())
    else: R = np.array(np.identity(3))
    nt = nk*nk # total number of points
    ik = 0
    from . import parallel
    if verbose>0: tr = timing.Testimator("BERRY CURVATURE",maxite=len(ks))
    if mode=="Green": # build the generator once, not per k-point
        f = h.get_gk_gen(delta=delta) # get generator
        def gI(e=0.,k=[0.,0.,0.]): return f(e=e,k=k,inv=True) # exact inverse
    def fp(ki): # function to compute the Berry curvature
        if parallel.cores == 1:
            if verbose>0: tr.iterate()
        else:
            if verbose>0:  print("Doing",ki)
        k = R@ki # change of basis
        if mode=="Wilson":
           b = berry_curvature(h,k,dk=dk,window=window,max_waves=max_waves)
        elif mode=="Green":
           b = berry_green(f,k=k,operator=operator,gI=gI)
        else:
          raise ValueError("unknown mode for the Berry curvature; the "
                  "accepted ones are 'Wilson' and 'Green'")
        return b
    from .topologytk.berry import use_berry_curvature_mesh
    if use_berry_curvature_mesh(h,mode=mode,window=window,max_waves=max_waves):
        # batched, numba-parallel path -- no interprocess dispatch
        from .topologytk.berry import berry_curvature_mesh
        bs = berry_curvature_mesh(h,np.array([R@ki for ki in ks]),dk=dk)
    else: # per-kpoint dispatch
        bs = parallel.pcall(fp,ks) # compute all the Berry curvatures
    if write: # write result in a file
        fo = open("BERRY_MAP.OUT","w") # open file
        for (b,k) in zip(bs,ks): # write everything
            fo.write(str(k[0])+"   "+str(k[1])+"     "+str(b)+"\n")
            fo.flush()
        fo.close() # close file
    return [ks[:,0],ks[:,1],np.array(bs)] # return result


berry_map = get_berry_curvature # alias

from .topologytk.wannier import smooth_gauge


def z2_wannier_centers(h,full=False,**kwargs):
    """Return the Wannier centers for the Z2 invariant, over half of the
    Brillouin zone, or over all of it with full=True"""
    # full used to be dropped here, so full=True returned half the flow
    return wannier_centers(h,full=full,**kwargs)


z2_vanderbilt = z2_wannier_centers # for compatibility


def wannier_centers(h,nk=30,nt=100,nocc=None,full=False,loop=0,pump=1,
        kfix=0.,gauge="lattice"):
    """Flow of the hybrid Wannier centers (Soluyanov-Vanderbilt algorithm)

    Returns an array whose first row is the momentum t along the reciprocal
    direction pump and whose other rows are the phases of the eigenvalues
    of the Wilson loop along the direction loop, one row per occupied band.
    t runs over half of the Brillouin zone, from t=0 to t=1/2, as the Z2
    invariant needs, or over all of it with full=True.

    loop, pump: indices (0, 1, or 2 in three dimensions) of the reciprocal
        lattice vectors the Wilson loop and the pumping run along
    kfix: in three dimensions, the momentum along the remaining direction,
        so that the flow is the one of the plane at that momentum
    gauge: "lattice" places every orbital at the origin of its cell, as
        pyqula's Bloch Hamiltonian does, which leaves every winding as it
        is; "atomic" places it at its position, so that the centers are
        positions (a phase 2 pi x for a center at the fractional
        coordinate x along loop), which is what a polarization needs
    nocc: the number of bands, counted from the lowest, whose centers are
        followed; by default the ones below zero energy. They have to be
        separated from the band above them by a gap everywhere"""
    from .topologytk.qgt import _check_gauge,_orbital_fractions
    _check_gauge(gauge)
    dim = h.dimensionality
    if dim not in (2,3):
        raise ValueError("the Wannier-center flow needs a two- or "
          +"three-dimensional Hamiltonian, and this one has dimensionality "
          +str(dim))
    if loop==pump or not (0<=loop<dim and 0<=pump<dim):
        raise ValueError("loop and pump must be two different directions "
          +"out of "+str(list(range(dim)))+"; got loop="+str(loop)
          +" and pump="+str(pump))
    if dim==2 and kfix!=0.:
        raise ValueError("kfix fixes the third momentum of a "
          +"three-dimensional Hamiltonian, and this one is two-dimensional")
    def kvector(k,t): # fractional momentum at loop momentum k and pump t
        kv = np.zeros(3)
        if dim==3: kv[3-loop-pump] = kfix # the remaining direction
        kv[loop] = k ; kv[pump] = t
        return kv
    out = [] # output list
    path = np.linspace(0.,1.,nk) # set of kpoints
    if full:  ts = np.linspace(0.,1.0,nt,endpoint=False)
    else:  ts = np.linspace(0.,0.5,nt) # from t=0 to t=1/2, both included
    hkgen = h.get_hk_gen() # Bloch Hamiltonian generator
    wfall = [[occupied_states(hkgen,kvector(k,t),nocc=nocc) for k in path]
                for t in ts]
    sizes = set([len(wf) for wft in wfall for wf in wft]) # occupied states
    if len(sizes)!=1:
        # the Wilson loop needs the same number of occupied states at every
        # k-point, and a ragged set used to die in np.array with an opaque
        # inhomogeneous-shape error
        raise ValueError("the number of occupied states changes over the "
          +"Brillouin zone (between "+str(min(sizes))+" and "+str(max(sizes))
          +"), so the occupied manifold is not separated by a gap and its "
          +"Wannier centers are not defined. Shift the Fermi energy into a "
          +"gap with h.shift_fermi / h.set_filling first")
    if gauge=="atomic": # u(k) -> exp(-i 2pi k.x) u(k) on each orbital
        frac = _orbital_fractions(h,h.intra.shape[0]) # (orbitals,dim)
        for (t,wft) in zip(ts,wfall):
            for (ik,k) in enumerate(path):
                phase = np.exp(2j*np.pi*frac@kvector(k,t)[:dim])
                wft[ik] = wft[ik]*phase[None,:] # rows are conjugated states
    fo = open("WANNIER_CENTERS.OUT","w")
    # select a continuos gauge for the first wave
    for it in range(len(ts)-1): # loop over ts
      wfall[it+1][0] = smooth_gauge(wfall[it][0],wfall[it+1][0]) 
    for it in range(len(ts)): # loop over t points
      row = [] # empty list for this row
      t = ts[it] # select the t point
      wfs = wfall[it] # get set of waves 
      for i in range(len(wfs)-1):
        wfs[i+1] = smooth_gauge(wfs[i],wfs[i+1]) # transform into a smooth gauge
      wf0 = wfs[0] # the loop closes on the first states, or in the atomic
      if gauge=="atomic": # gauge on their periodic image exp(-i 2pi x) u(0)
          wf0 = wf0*np.exp(2j*np.pi*frac[:,loop])[None,:]
      m = uij(wf0,wfs[len(wfs)-1]) # matrix of wavefunctions
      evals = lg.eigvals(m) # eigenvalues of the rotation 
      x = -np.angle(evals) # m is the conjugate of the Wilson loop, so the
                           # centers 2 pi x are minus the phases of m
      fo.write(str(t)+"    ") # write pumping variable
      row.append(t) # store
      for ix in x: # loop over phases
        fo.write(str(ix)+"  ")
        row.append(ix) # store
      fo.write("\n")
      out.append(row) # store
    fo.close()
    return np.array(out).transpose() # transpose the map


def z2_invariant(h,nk=60,nt=60,nocc=None):
  """Compute Z2 invariant with pumping of Wannier centers"""
  return z2_wannier_winding(h,nk=nk,nt=nt,nocc=nocc)




# integration= of chern() -> the routine that evaluates the Chern number
_chern_integrations = {
    "grid": lambda h,**kw: mesh_chern(h,**kw),
    "qtci": lambda h,**kw: chern_qtci(h,**kw),
    "wannier": lambda h,**kw: wannier_winding(h,full=True,**kw),
    }


def chern(h,integration="grid",**kwargs):
    """Compute Chern invariant.

    integration: "grid" (default) sums the Berry curvature over a fixed
    k-point mesh, see mesh_chern. "qtci" instead integrates the Berry
    curvature over the BZ using qutecipy (tensor cross interpolation +
    Gauss-Kronrod quadrature), see chern_qtci, adaptively refining the
    sampling instead of relying on a fixed mesh density. "wannier" counts
    the winding of the hybrid Wannier centers, see wannier_winding.

    An operator= weights the Berry curvature by an operator, see
    mesh_chern. On a Nambu Hamiltonian the name "sz" is the physical spin,
    sigma_z tau_0 in the basis (c_up, c_dn, c^dag_dn, -c^dag_up), which an
    equal-spin pairing does not conserve: the two decoupled sectors of a
    helical p-wave superconductor are labelled by sigma_z tau_z,
    diag(1,-1,-1,1), and with "sz" the weighted Chern number is zero there.
    """
    if integration not in _chern_integrations:
        raise ValueError("unknown integration '"+str(integration)+"' for "
          +"the Chern number; it must be one of "
          +str(list(_chern_integrations)))
    return _chern_integrations[integration](h,**kwargs)


def wannier_winding(h,nk=30,nt=100,full=True,loop=0,pump=1,kfix=0.,
        nocc=None):
    """Signed number of times the hybrid Wannier centers cross a fixed line

    At each momentum t along the second reciprocal direction, the Wilson
    loop along the first one has the hybrid Wannier centers of the occupied
    bands as the phases of its eigenvalues (wannier_centers). We follow
    them as t runs over the whole Brillouin zone (full=True), or over half
    of it, from the time-reversal-invariant momentum t=0 to t=1/2
    (full=False), and count how many times they cross a horizontal line
    theta0, downwards minus upwards. Over the whole zone the count is the
    Chern number, the same for any line: the Berry flux through the strip
    between two consecutive t is minus the change of the Berry phase along
    the loop, so the sum of the centers moves by -C lattice constants per
    cycle with the orientation of h.get_chern(). Over half of it, its parity is
    the Z2 invariant of a time-reversal-symmetric system, since Kramers
    partners that switch cross any such line an odd number of times
    (Soluyanov-Vanderbilt, arXiv:1102.5600; Z2Pack, arXiv:1610.08983); see
    z2_wannier_winding.

    nk: k-points of each Wilson loop
    nt: values of t
    full: the whole Brillouin zone (the Chern number) or half of it (the
        count whose parity is the Z2 invariant)
    loop, pump, kfix: the directions and the plane, as for wannier_centers;
        exchanging loop and pump reverses the orientation, and so the sign
        of the Chern number
    nocc: the number of bands counted from the lowest, as for
        wannier_centers

    The count needs no tracking of the individual centers. Between two
    consecutive t the sum phi of the centers moves by dphi, brought into
    [-pi,pi), and the sum F of the centers measured from theta0 in
    [0,2pi) moves by the same amount, except that it drops by 2pi each time
    a center crosses theta0 upwards and gains 2pi each time one crosses it
    downwards. The count is therefore (F(end) - F(start) - sum of dphi)/2pi,
    and on the closed loop of full=True the F terms cancel, which leaves the
    winding of phi. theta0 goes in the middle of the largest gap between the
    centers at the two ends, where they are furthest from it. The result is
    an integer at any nt, and the right one as long as the sum of the
    centers moves by less than half a lattice constant per step; the
    centers move fast where the gap is small, so a small gap needs a larger
    nt."""
    m = wannier_centers(h,nk=nk,nt=nt,full=full,loop=loop,pump=pump,
            kfix=kfix,nocc=nocc)
    x = m[1:] # centers, one row per occupied band and one column per t
    if full: x = np.concatenate([x,x[:,:1]],axis=1) # close the loop in t
    phi = np.sum(x,axis=0) # sum of the centers at each t
    dphi = (np.diff(phi)+np.pi)%(2.*np.pi) - np.pi # each step into [-pi,pi)
    theta0 = _largest_gap_center(np.concatenate([x[:,0],x[:,-1]]))
    def F(xt): return np.sum((xt-theta0)%(2.*np.pi)) # measured from theta0
    c = (F(x[:,-1]) - F(x[:,0]) - np.sum(dphi))/(2.*np.pi) # down minus up
    return int(np.round(c))


def _largest_gap_center(thetas):
    """Middle of the largest gap between a set of phases on the circle"""
    t = np.sort(np.mod(thetas,2.*np.pi))
    gaps = np.diff(np.concatenate([t,[t[0]+2.*np.pi]])) # the last one wraps
    i = np.argmax(gaps)
    return t[i] + gaps[i]/2.


def z2_wannier_winding(h,nk=100,nt=100,nocc=None,**kwargs):
    """Z2 invariant from the Wannier-center flow over half of the Brillouin
    zone: +1 trivial, -1 topological, the parity of wannier_winding with
    full=False (loop, pump and kfix select the plane, as there)"""
    return 1 - 2*(wannier_winding(h,nk=nk,nt=nt,full=False,nocc=nocc,
                **kwargs)%2)


def z2_invariant_3d(h,nk=60,nt=60):
    """Strong and weak Z2 indices nu0;(nu1 nu2 nu3) of a three-dimensional
    time-reversal-symmetric insulator, each 0 or 1

    The plane k_i=1/2 of the Brillouin zone is time-reversal symmetric and
    has a Z2 invariant of its own, which is the weak index nu_i, and the
    strong index nu0 is the product of the invariants of the planes k_i=0
    and k_i=1/2, which has to be the same for the three directions i
    (Fu-Kane-Mele, cond-mat/0607699; Soluyanov-Vanderbilt, arXiv:1102.5600).
    Each plane goes through z2_wannier_winding, six planes in all; the
    weak indices are the components of G = nu1 b1 + nu2 b2 + nu3 b3 in the
    reciprocal lattice vectors of the geometry.

    nk, nt: k-points of each Wilson loop and momenta of each half-zone flow

    If the three strong indices disagree the flow was not resolved, or the
    system has no gap, and this raises rather than picking one."""
    if h.dimensionality!=3:
        raise ValueError("the strong and weak Z2 indices need a "
          +"three-dimensional Hamiltonian, and this one has dimensionality "
          +str(h.dimensionality)+"; use z2_invariant in two dimensions")
    z = np.zeros((3,2),dtype=int) # 1 where the plane k_i=0, 1/2 is odd
    for i in range(3):
        (loop,pump) = [a for a in range(3) if a!=i] # the plane of k_i fixed
        for (j,kfix) in enumerate([0.,0.5]):
            z[i,j] = wannier_winding(h,nk=nk,nt=nt,full=False,loop=loop,
                    pump=pump,kfix=kfix)%2
    nu0s = (z[:,0]+z[:,1])%2 # the strong index from each direction
    if len(set(nu0s))!=1:
        raise ValueError("the strong Z2 index comes out different from the "
          +"three directions ("+str(list(nu0s))+"), so the Wannier-center "
          +"flow of some plane was not resolved or the gap closes; raise nk "
          +"and nt, and check that the system is gapped")
    return (int(nu0s[0]),tuple(int(v) for v in z[:,1]))


def chern_vector(h,nk=30,nt=100,kfix=0.):
    """Chern numbers (C1,C2,C3) of the planes k_1, k_2 and k_3 fixed at kfix
    of a three-dimensional insulator, from the winding of the hybrid Wannier
    centers

    C_i is the Chern number of the plane spanned by the next two reciprocal
    lattice vectors in cyclic order, the Wilson loop along b_{i+1} and the
    pumping along b_{i+2}, so that C3 is the Chern number of a layer in the
    (b1,b2) plane with the orientation of the two-dimensional h.get_chern().
    In an insulator it does not depend on kfix; the Hall conductivity of a
    layered system is set by these three integers."""
    if h.dimensionality!=3:
        raise ValueError("the Chern vector needs a three-dimensional "
          +"Hamiltonian, and this one has dimensionality "
          +str(h.dimensionality)+"; use h.get_chern() in two dimensions")
    return tuple(wannier_winding(h,nk=nk,nt=nt,full=True,loop=(i+1)%3,
                    pump=(i+2)%3,kfix=kfix) for i in range(3))



from .topologytk.nestedwilson import wannier_sector_polarization
from .topologytk.nestedwilson import wannier_gap
from .topologytk.nestedwilson import quadrupole_moment


def operator_berry(hin,k=[0.,0.],operator=None,delta=0.00001,ewindow=None):
    """Calculates the Berry curvature using an arbitrary operator

    ewindow: None, or a callable taking a band energy and returning
      whether to keep that band, exactly as on bandstructure.get_bands. It
      narrows the occupied (E<=0) manifold the curvature is summed over;
      it does not replace the occupancy condition, so a window covering
      the whole spectrum reproduces the unwindowed value.
    """
    k = np.array(list(k) + [0.]*(3-len(k))) # multicell.derivative needs 3 components
    h = multicell.turn_multicell(hin) # turn to multicell form
    dhdx = multicell.derivative(h,k,order=[1,0]) # derivative
    dhdy = multicell.derivative(h,k,order=[0,1]) # derivative
    hkgen = h.get_hk_gen() # get generator
    hk = hkgen(k) # get hamiltonian
    (es,ws) = algebra.eigh(hkgen(k)) # initial waves
    ws = np.conjugate(np.transpose(ws)) # transpose the waves
    n = len(es) # number of energies
    if operator is None: operator = np.identity(dhdx.shape[0],dtype=np.complex128)
    if ewindow is None: # every occupied band contributes
        from .topologytk.operatorberry import berry_curvature as bc90
        b = bc90(dhdx,dhdy,ws,es,operator,delta) # berry curvature
    else: # restrict the sum to the occupied bands inside the window
        if not callable(ewindow):
            raise TypeError("ewindow must be a callable taking a band energy "
                    "and returning whether to keep that band, and not a "
                    +str(type(ewindow)))
        from .topologytk.operatorberry import berry_curvature_bands as bcb90
        bs = bcb90(dhdx,dhdy,ws,es,operator,delta) # one value per band
        keep = np.array([e<=0. and bool(ewindow(e)) for e in es])
        b = np.sum(bs[keep]) # only the selected bands
    return b*np.pi*np.pi*8 # normalize so the sum is 2pi Chern



def operator_berry_bands(hin,k=[0.,0.],operator=None,delta=0.00001):
    """Calculates the Berry curvature using an arbitrary operator"""
    k = np.array(list(k) + [0.]*(3-len(k))) # multicell.derivative needs 3 components
    h = multicell.turn_multicell(hin) # turn to multicell form
    dhdx = multicell.derivative(h,k,order=[1,0]) # derivative
    dhdy = multicell.derivative(h,k,order=[0,1]) # derivative
    hkgen = h.get_hk_gen() # get generator
    hk = hkgen(k) # get hamiltonian
    (es,ws) = algebra.eigh(hkgen(k)) # initial waves
    ws = np.conjugate(np.transpose(ws)) # transpose the waves
    from .topologytk.operatorberry import berry_curvature_bands as bcb90
    if operator is None: operator = np.identity(dhdx.shape[0],dtype=np.complex128)
    bs = bcb90(dhdx,dhdy,ws,es,operator,delta) # berry curvatures
    return (es,bs*np.pi*np.pi*8) # normalize so the sum is 2pi Chern




from .topologytk.qgt import quantum_geometric_tensor_k
from .topologytk.qgt import quantum_geometric_tensor_path
from .topologytk.qgt import quantum_geometric_tensor_mesh
from .topologytk.qgt import quantum_metric_from_qgt
from .topologytk.qgt import berry_curvature_from_qgt
from .topologytk.qgt import chern_from_qgt


def quantum_geometric_tensor(h,**kwargs):
    """Multiband (non-Abelian) quantum geometric tensor of a multiorbital
    Hamiltonian, see topologytk/qgt.py for the formula and references"""
    return quantum_geometric_tensor_k(h,**kwargs)


def quantum_metric(h,**kwargs):
    """Quantum metric (symmetric part of the quantum geometric tensor)
    of a multiorbital Hamiltonian, at a single k-point"""
    non_abelian = kwargs.get("non_abelian",False)
    Q = quantum_geometric_tensor_k(h,**kwargs)
    return quantum_metric_from_qgt(Q,non_abelian=non_abelian)







def spin_chern(h,nk=40,delta=0.00001,k0=[0.,0.],expandk=1.0):
  """Calculate the spin Chern number"""
  kxs = np.linspace(-.5,.5,nk,endpoint=False)*expandk + k0[0]
  kys = np.linspace(-.5,.5,nk,endpoint=False)*expandk + k0[1]
  kk = [] # list of kpoints
  for i in kxs:
    for j in kys:
      kk.append(np.array([i,j])) # store vector
  sz = operators.get_sz(h) # get sz operator
  bs = [operator_berry(h,k=ki,operator=sz,delta=delta) for ki in kk] # get all berries
  fo = open("BERRY_CURVATURE_SZ.OUT","w") # open file
  for (k,b) in zip(kk,bs):
    fo.write(str(k[0])+"   ")
    fo.write(str(k[1])+"   ")
    fo.write(str(b)+"\n")
  fo.close() # close file
  bs = np.array(bs)/(2.*np.pi) # normalize by 2 pi
  return sum(bs)/len(kk)


def write_spin_berry(h,kpath,delta=0.00001,operator=None):
  """Calculate and write in file the Berry curvature"""
  if operator is None: sz = operators.get_sz(h) # get sz operator
  else: sz = operator # assign operator
  be = [operator_berry(h,k=k,operator=sz,delta=delta) for k in kpath] 
  fo = open("BERRY_CURVATURE_SZ.OUT","w") # open file
  for (k,b) in zip(kpath,be):
    fo.write(str(k[0])+"   ")
    fo.write(str(k[1])+"   ")
    fo.write(str(b)+"\n")
  fo.close() # close file






def precise_spin_chern(h,delta=0.00001,tol=0.1,nk=None):
  """ Calculates the chern number of a 2d system """
  from scipy import integrate
  if nk is not None: # same contract as precise_chern, see the note there
      raise ValueError("precise_spin_chern integrates the Brillouin zone "
          "adaptively (scipy.integrate.dblquad), so it has no k-mesh and "
          "nk is meaningless here; got nk="+str(nk)+". Use tol to set the "
          "integration tolerance, or topology.spin_chern(h,nk=...) for the "
          "same s_z-weighted integral on a fixed mesh")
  err = {"epsabs" : 0.01, "epsrel": 0.01,"limit" : 20}
  sz = operators.get_sz(h) # get sz operator
  def f(x,y): # function to integrate
    return operator_berry(h,np.array([x,y]),delta=delta,operator=sz)
  c = integrate.dblquad(f,0.,1.,lambda x : 0., lambda x: 1.,epsabs=tol,
                          epsrel=tol)
  return c[0]/(2.*np.pi)


from .topologytk.green import berry_green_generator
from .topologytk.green import berry_green
from .topologytk.green import berry_operator


from .topologytk.green import berry_green_rmap_kpoint



def spatial_berry_density(h,**kwargs):
    """Berry density at the Fermi energy and spatially resolved"""
    return Omega_rmap(h,integral=False,**kwargs)



def Omega_rmap(h,nrep=5,k=[0.,0.,0.],operator=None,nk=None,**kwargs):
  """
  Write the spatial resolved Berry curvature of a kpoint in a file.
  If nk is provided, it does a sum over reciprocal space
  """
  if operator is not None: # this is a dirty workaround
    if type(operator) is str:
      operator = h.get_operator(operator) 
    else: pass
  if nk is None: # kpoint given
    out = berry_green_rmap_kpoint(h,k=k,operator=operator,**kwargs) 
  else: # kpoint not given
    if operator is not None: 
        from . import gauge
        operator = gauge.Operator2canonical_gauge(h,operator)
        print("Fixing the gauge in the operator")
    ks = klist.kmesh(h.dimensionality,nk=nk) # kpoints
    def f(ki):
        print("kpoint",ki)
        return berry_green_rmap_kpoint(h,k=ki,operator=operator,**kwargs)
    out = parallel.pcall(f,ks) # compute all
    out = np.mean(out,axis=0) # resum
  from . import geometry
  from .ldos import spatial_dos
  # write in a file
  geometry.write_profile(h.geometry,
          spatial_dos(h,out),name="BERRY_RMAP.OUT",nrep=nrep)
  return out



# alias for compatibility
from .topologytk.green import dOmega_dE
from .topologytk.green import dOmega_dE_generator


def dOmega_dE_kmap(h,nk=40,reciprocal=True,nsuper=1,
               delta=None,operator=None,dk=0.01):
  """Compute a Berry density map dOmega/dE (k) at a fixed energy"""
  if delta is None: delta = 5./nk
  if reciprocal: R = h.geometry.get_k2K()
  else: R = np.array(np.identity(3))
  fo = open("BERRY_DENSITY_KMAP.OUT","w") # open file
  nt = nk*nk # total number of points
  ik = 0
  ks = [] # list with kpoints
  for x in np.linspace(-nsuper,nsuper,nk,endpoint=False):
    for y in np.linspace(-nsuper,nsuper,nk,endpoint=False):
        ks.append([x,y,0.])
  tr = timing.Testimator("BERRY DENSITY",maxite=len(ks))
  def fp(ki): # function to compute the Berry curvature
      if parallel.cores == 1: tr.iterate()
      else: print("Doing",ki)
      r = np.array(ki) # real space vectors
      k = R@r # change of basis
      b = dOmega_dE(h,k=k,operator=operator,dk=dk) # get the density
      return b
  bs = parallel.pcall(fp,ks) # compute all the Berry curvatures
  for (b,k) in zip(bs,ks): # write everything
      fo.write(str(k[0])+"   "+str(k[1])+"     "+str(b)+"\n")
      fo.flush()
  fo.close() # close file



def chern_density(h,nk=10,operator=None,delta=0.02,dk=0.02,
        write=False,
        es=np.linspace(-1.0,1.0,40)):
  """Compute the Chern density as a function of the energy"""
  operator = get_operator(h,operator) # accept a name, matrix or callable
  ks = klist.kmesh(h.dimensionality,nk=nk)
  cs = np.zeros(es.shape[0]) # initialize
  # dOmega_dE_generator wraps the same berry_green_generator call this used
  # to duplicate inline (with the correct sign convention -- see its
  # docstring/comments -- and the exact, rather than averaged, Green's
  # function inverse)
  fdomega = dOmega_dE_generator(h,operator=operator,delta=delta,dk=dk)
  tr = timing.Testimator("CHERN DENSITY",maxite=len(ks))
  from . import parallel
  def fp(k): # compute berry curvatures
    if parallel.cores==1: tr.iterate()
    else: print(k)
#    k = np.random.random(3)
    return np.array([fdomega(k=k,e=e) for e in es]) # dOmega/dE at all energies
  out = parallel.pcall(fp,ks) # compute everything
  for o in out: cs += o # add contributions
  cs = cs/(len(ks)*np.pi*2) # normalize
  from scipy.integrate import cumulative_trapezoid
  csi = cumulative_trapezoid(cs,x=es,initial=0) # integrate
  if write:
      np.savetxt("CHERN_DENSITY.OUT",np.array([es,cs]).T)
      np.savetxt("CHERN_DENSITY_INTEGRATED.OUT",np.array([es,csi]).T)
  return (es,cs,csi)








# The zero-temperature intrinsic Hall conductivity in units of e^2/h is the
# Chern number, so this is an alias and not a separate routine -- it takes
# chern's arguments (nk, integration, operator, ...). There used to be a
# second, Monte-Carlo definition of this name earlier in the module, which
# this binding shadowed: the dk/n keywords it advertised had no effect on
# anything, since the name always resolved to chern at import time.
hall_conductivity = chern




def get_operator(h,op):
    """ Wrapper for operators """
    if op is None: return None
    if type(op)==str: # string
        # the valley operator is the only one that takes projector=True;
        # every other name goes through the ordinary Hamiltonian lookup.
        # Both branches of this test used to return the valley operator,
        # so asking for a "sz"-resolved Berry curvature silently computed
        # the valley-projected one -- and any unrecognised string became
        # "valley" instead of raising.
        if op=="valley": return h.get_operator("valley",projector=True)
        else: return h.get_operator(op)
    if callable(op): return op # function
    # np.array is a function, not a type, so `type(op)==np.array` was never
    # true and a raw matrix fell off the end of the function as None, i.e.
    # unprojected
    if algebra.ismatrix(op): return h.get_operator(op)
    raise TypeError("unrecognised operator of type "+str(type(op))
        +", expected None, a string name, a matrix, an Operator "
        +"or a callable")



from .topologytk import realspace

real_space_chern = realspace.real_space_chern


from .topologytk.topologicalsector import get_berry_curvature_operator_sector
from .topologytk.topologicalsector import get_chern_operator_sector
