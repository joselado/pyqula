
from ..geometry import same_site
import numpy as np
import scipy.sparse as sp
from ..utilities import get_callable
from ..check import require_sublattice
from .dvector import dvector2delta


# The pairing symmetries live in a registry (mode -> builder) rather than in
# an if/elif chain, so pairing_modes below is derived from the dispatch
# instead of being a second tuple kept in sync by hand, and adding a channel
# is one dict entry. Every builder takes the Hamiltonian, the d-vector
# callable and the caller's keyword bag, and returns the weight function
# weightf(r1,r2) -> 2x2 pairing matrix.
# (tests/superconductivity/test_pairing_modes.py builds every advertised one)
#
# Every weight has to obey Fermi antisymmetry, which in pyqula's Nambu basis
# (c_up, c_dn, c_dn^dag, -c_up^dag) reads D_ij = sigma_y D_ji^T sigma_y: the
# spin-singlet part even under i <-> j and the d-vector odd. A weight that
# breaks it is not a pairing at all, since the part of the BdG matrix that
# violates it adds only a constant to the many-body Hamiltonian, and yet it
# shows up in the BdG spectrum. The "haldane"/"antihaldane" (odd singlet),
# "swavez" (onsite triplet) and "SnnAB" (even triplet off a bipartite
# lattice) modes did this and were removed.


def _on_sublattice(H,mode,weightf):
    """Sublattice-resolved weights need a labeled sublattice. Without one
    they added zero pairing without saying so (square lattice) or died in
    get_index with an IndexError or an AttributeError"""
    require_sublattice(H,"the '"+mode+"' pairing")
    return weightf


# mode -> builder(H,df,kwargs) -> weightf(r1,r2). The helpers the builders
# call are defined further down the module; that is fine, every builder is
# a lambda and resolves them when it is called, not when it is defined.
_pairing_builders = {
  "swave": lambda H,df,kw: lambda r1,r2: swave(r1,r2,H=H,**kw),
  "extended_swave": lambda H,df,kw: lambda r1,r2: swave(r1,r2,H=H,nn=1,**kw),
  "triplet": lambda H,df,kw: lambda r1,r2: pwave(r1,r2,df,**kw),
  "pwave": lambda H,df,kw: lambda r1,r2: pwave(r1,r2,df,**kw),
  "nodal_fwave": lambda H,df,kw: nodal_fwave_generator(df,H=H,**kw),
  "chiral_pwave": lambda H,df,kw: lambda r1,r2: get_triplet(r1,r2,df,L=1,**kw),
  "chiral_fwave": lambda H,df,kw: get_triplet_generator(df,L=3,H=H,**kw),
  "chiral_dwave": lambda H,df,kw: lambda r1,r2: get_singlet(r1,r2,L=2,**kw),
  "chiral_gwave": lambda H,df,kw: lambda r1,r2: get_singlet(r1,r2,L=4,**kw),
  "px": lambda H,df,kw: lambda r1,r2: px(r1,r2),
  "dpid": lambda H,df,kw: lambda r1,r2: dpid(r1,r2,**kw),
  "swaveA": lambda H,df,kw: _on_sublattice(H,"swaveA",
          lambda r1,r2: swaveA(H.geometry,r1,r2)),
  "swaveB": lambda H,df,kw: _on_sublattice(H,"swaveB",
          lambda r1,r2: swaveB(H.geometry,r1,r2)),
  "swavesublattice": lambda H,df,kw: _on_sublattice(H,"swavesublattice",
          lambda r1,r2: swaveB(H.geometry,r1,r2) - swaveA(H.geometry,r1,r2)),
  "dx2y2": lambda H,df,kw: lambda r1,r2: dx2y2(r1,r2,H=H,**kw),
  "nodal_dwave": lambda H,df,kw: lambda r1,r2: dx2y2(r1,r2,H=H,**kw),
  "dxy": lambda H,df,kw: lambda r1,r2: dxy(r1,r2,H=H,**kw),
  "snn": lambda H,df,kw: lambda r1,r2: swavenn(r1,r2),
  "C3nn": lambda H,df,kw: lambda r1,r2: C3nn(r1,r2),
  }


# derived from the dispatch, never a second list kept in sync by hand
pairing_modes = tuple(_pairing_builders)


def get_pairing_modes():
    """Return every pairing symmetry that pairing_generator accepts"""
    return tuple(_pairing_builders)


def pairing_generator(self,delta=0.0,mode="swave",d=[0.,0.,1.],
    **kwargs):
    """Create a generator, taking as input two positions, and returning
    the 2x2 pairing matrix"""
    # wrapper for the amplitude and d-vector
    deltaf = get_callable(delta) # callable for the amplitude
    df = get_callable(d) # callable for the d-vector
    if callable(mode):
        weightf = mode # mode is a function returning a 2x2 pairing matrix
    else:
        if mode not in _pairing_builders:
            raise ValueError("unknown pairing mode '"+str(mode)+"'; it must be "
              +"one of "+str(list(pairing_modes))+", or a callable returning "
              +"the 2x2 pairing matrix")
        weightf = _pairing_builders[mode](self,df,kwargs)
    matrixf = lambda r1,r2: deltaf((r1+r2)/2.)*weightf(r1,r2) 
    return matrixf # return function


def check_fermi_antisymmetry(h,weightf,tol=1e-6):
    """Raise if a pairing weight breaks Fermi antisymmetry,
    D(r1,r2) = sigma_y D(r2,r1)^T sigma_y, on the pairs of positions that
    add_pairing evaluates: every pair inside the cell, and every pair
    between the cell and each of its neighboring replicas. The registered
    modes obey it by construction, so this is only needed for a callable"""
    sy = np.array([[0.,-1j],[1j,0.]])
    r = h.geometry.r
    replicas = [r] # the cell itself first, then its neighbors
    for d in h.geometry.neighbor_directions():
        if np.dot(d,d)>1e-4: replicas.append(h.geometry.replicas(d=d))
    for r2 in replicas:
        for r1i in r:
            for r2j in r2:
                w12 = np.array(weightf(r1i,r2j),dtype=np.complex128)
                w21 = np.array(weightf(r2j,r1i),dtype=np.complex128)
                if w12.shape!=(2,2):
                    raise ValueError("a pairing callable must return the "
                        "2x2 pairing matrix, and this one returned shape "
                        +str(w12.shape))
                dev = np.max(np.abs(w12 - sy@w21.T@sy))
                if dev>tol:
                    raise ValueError("the pairing callable breaks Fermi "
                        "antisymmetry, which needs D(r1,r2) = "
                        "sigma_y D(r2,r1)^T sigma_y (the singlet part even "
                        "under r1 <-> r2 and the d-vector odd): at r1="
                        +str(np.round(r1i,4))+", r2="+str(np.round(r2j,4))
                        +" the two sides differ by "+str(dev)+". The part "
                        "that breaks it is not a pairing, it only adds a "
                        "constant to the many-body Hamiltonian, and yet it "
                        "shows up in the BdG spectrum")


def check_periodic_pairing(blocks,callables=(),tol=1e-6):
    """Raise unless the electron-hole blocks D_R that add_pairing built,
    between the cell and its replica at R (a dict R -> 2n x 2n block, R=0
    included), obey Fermi antisymmetry across cells,

        D_R = S D_{-R}^T S,   S = 1 x sigma_y.

    The pairing between site i of the cell and site j of the cell at R, and
    the one between site j of the cell and site i of the cell at -R, are
    the same bond seen from its two ends, but add_pairing evaluates them at
    two positions a lattice vector apart. A d-vector, amplitude or weight
    given as a function of position that is not periodic with the lattice
    gives them two different values, and the part of the BdG matrix that
    breaks the relation is not a pairing at all: it only adds a constant to
    the many-body Hamiltonian, and yet it shows up in the BdG spectrum and
    in the d-vector non-unitarity. callables names the arguments that were
    given as functions, for the message"""
    sy = np.array([[0.,-1j],[1j,0.]])
    n = blocks[(0,0,0)].shape[0]//2 # number of sites
    S = sp.kron(sp.identity(n),sy,format="csr")
    scale = max([abs(b).max() if b.nnz>0 else 0. for b in blocks.values()])
    if scale==0.: return # no pairing at all
    for R in blocks:
        mR = tuple(-np.array(R))
        dev = sp.coo_matrix(blocks[R] - S@blocks[mR].T@S)
        if dev.nnz==0: continue
        a = np.argmax(np.abs(dev.data))
        if np.abs(dev.data[a])<=tol*scale: continue
        i = dev.row[a]//2 ; j = dev.col[a]//2 # the two sites
        if len(callables)>0:
            given = " and ".join(callables)+(" was" if len(callables)==1
                        else " were")+" given as a function, and"
        else: given = ""
        raise ValueError("the pairing breaks Fermi antisymmetry between "
            "unit cells, which needs a pairing given as a function of "
            "position to be periodic with the lattice: "+given+" the "
            "pairing between site "+str(i)+" of the cell and site "+str(j)
            +" of the cell at R="+str(tuple(int(x) for x in R))+" differs "
            "by "+str(np.round(np.abs(dev.data[a]),6))+" (of a largest "
            "pairing "+str(np.round(scale,6))+") from the one between site "
            +str(j)+" of the cell and site "+str(i)+" of the cell at -R, "
            "which is the same bond seen from its other end. Make it "
            "periodic, for instance on a supercell commensurate with the "
            "modulation, or use a zero-dimensional geometry")


# matrices for the e-h subsector
iden = np.array([[1.,0.],[0.,1.]],dtype=np.complex128)
tauz = np.array([[1.,0.],[0.,-1.]],dtype=np.complex128)
taux = np.array([[0.,1.],[1.,0.]],dtype=np.complex128)
tauy = np.array([[0.,1j],[-1j,0.]],dtype=np.complex128)
UU = taux + 1j*tauy # projector in the UU sector
DD = taux - 1j*tauy # projector in DD sector


def swavenn(r1,r2):
    dr = r1-r2
    dr2 = dr.dot(dr)
    if 0.99<dr2<1.001: # first neighbor
        return 1.0*iden
    return 0.0*iden


def dx2y2(r1,r2,**kwargs):
    """Function with first neighbor dx2y2 profile"""
    return (get_singlet(r1,r2,L=2,**kwargs) + get_singlet(r1,r2,L=-2,**kwargs))/2.
#    dr = r1-r2
#    dr2 = dr.dot(dr)
#    if 0.99<dr2<1.001: # first neighbor
#        return (dr[0]**2 - dr[1]**2)*iden
#    return 0.0*iden


def dxy(r1,r2,**kwargs):
    """Function with first neighbor dxy profile"""
    return (get_singlet(r1,r2,L=2,**kwargs) - get_singlet(r1,r2,L=-2,**kwargs))/2.
    dr = r1-r2
    dr2 = dr.dot(dr)
    if 0.99<dr2<1.001: # first neighbor
        return (dr[0]*dr[1])*iden
    return 0.0*iden


def swave(r1,r2,nn=0,**kwargs):
    """Function with first neighbor dxy profile"""
    if nn==0: return same_site(r1,r2)*np.identity(2)
    else: return get_singlet(r1,r2,L=0,nn=nn,**kwargs)


def C3nn(r1,r2):
    """Function with first neighbor C3 profile"""
    dr = r1-r2
    dr2 = dr.dot(dr)
    if 0.99<dr2<1.001: # first neighbor
#        return dr[0]
        phi = np.arctan2(dr[1],dr[0]) # angle
        return 1.0*np.exp(1j*phi)*tauz
    return 0.0*tauz







def swaveA(g,r1,r2):
    """Swave only in A"""
    dr = r1-r2
    dr2 = dr.dot(dr)
    if dr2<0.001: # first neighbor
        i = g.get_index(r1,replicas=False)
        if i is None: return 0.0
        if g.sublattice[i]==1:
          return 1.0*iden
    return 0.0*iden


def swaveB(g,r1,r2):
    """Swave only in A"""
    dr = r1-r2
    dr2 = dr.dot(dr)
    if dr2<0.001: # first neighbor
        i = g.get_index(r1,replicas=False)
        if i is None: return 0.0
        if g.sublattice[i]==-1:
          return 1.0*iden
    return 0.0*iden


def px(r1,r2):
    """Function with first neighbor px profile"""
    dr = r1-r2 ; dr2 = dr.dot(dr)
    if 0.99<dr2<1.001: return dr[0]*tauz
    return 0.0*tauz



def get_triplet_generator(df,nn=1,H=None,**kwargs):
    if nn>1: # more than first neighbor
      dist = H.geometry.get_neighbor_distances(n=nn)[nn-1] # get this distance
      dist2 = dist**2
    else: dist2 = 1.0 # first neighbor
    return lambda r1,r2: get_triplet(r1,r2,df,dist2=dist2,**kwargs)



def get_triplet(r1,r2,df,L=1,dist2=1.0):
    """Function for triplet order"""
    dr = r1-r2 ; dr2 = dr.dot(dr)
    if abs(L)%2!=1:
        raise ValueError("a triplet order parameter needs an odd angular "
                "momentum L")
    if np.abs(dr2-dist2)<1e-4:
        phi = np.arctan2(dr[1],dr[0])
        d = df((r1+r2)/2.) # evaluate dvector
        delta = dvector2delta(d) # compute the local deltas
        ms = np.array([[delta[2],delta[0]],[delta[1],-delta[2]]])
        return np.exp(1j*phi*L)*np.array(ms,dtype=np.complex128)
    else: return 0.0*tauz


def get_singlet(r1,r2,L=2,phi0=0.,H=None,nn=1):
    """Function for p-wave order"""
    if L%2!=0:
        raise ValueError("a singlet order parameter needs an even angular "
                "momentum L")
    dr = r1-r2 ; dr2 = dr.dot(dr)
    if nn>1: # more than first neighbor
      dist = H.geometry.get_neighbor_distances(n=nn)[nn-1] # get this distance
      dist2 = dist**2
    else: dist2 = 1.0 # first neighbor
#      print(dist2) ; exit()
    if np.abs(dr2-dist2)<1e-4:
        phi = np.arctan2(dr[1],dr[0])
        out = np.exp(1j*(phi+phi0*np.pi*2.)*L)
#        print(out)
        return out*iden
    else: return 0.0*tauz


def get_deltaud(r1,r2,f):
    """Return a Delta ud, given a certain function f of the two positions"""
    return iden*f(r1,r2)



def pwave(*args,**kwargs): 
    return get_triplet(*args,L=1,**kwargs)


def dpid(*args,**kwargs):
    return get_singlet(*args,L=2,**kwargs)

def nodal_fwave_generator(*args,dphi=0.,**kwargs): 
    """Generator for real f-wave order"""
    z = np.exp(1j*np.pi*2*dphi) # complex rotation
    f1 = get_triplet_generator(*args,L=3,**kwargs)
    f2 = get_triplet_generator(*args,L=-3,**kwargs)
    return lambda r1,r2: f1(r1,r2) + z*f2(r1,r2)

def nodal_fwave(*args,dphi=0.,**kwargs): 
    z = np.exp(1j*np.pi*2*dphi) # complex rotation
    return get_triplet(*args,L=3,**kwargs) + z*get_triplet(*args,L=-3,**kwargs)
