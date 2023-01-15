
from ..geometry import same_site
import numpy as np
from ..kanemele import get_haldane_function
from ..utilities import get_callable
from .dvector import dvector2delta


def pairing_generator(self,delta=0.0,mode="swave",d=[0.,0.,1.],
    **kwargs):
    """Create a generator, taking as input two positions, and returning
    the 2x2 pairing matrix"""
    # wrapper for the amplitude and d-vector
    deltaf = get_callable(delta) # callable for the amplitude
    df = get_callable(d) # callable for the d-vector
    if callable(mode):
        weightf = mode # mode is a function returning a 2x2 pairing matrix
    elif mode=="swave":
        weightf = lambda r1,r2: swave(r1,r2,H=self,**kwargs) 
    elif mode=="deltaud":
        # this is a workaround for delta_ud alone!
        weightf = lambda r1,r2: deltaud(r1,r2,deltaf) 
    elif mode=="extended_swave":
        weightf = lambda r1,r2: swave(r1,r2,H=self,nn=1,
                     **kwargs) #same_site(r1,r2)*np.identity(2)
    elif mode=="triplet": 
        weightf = lambda r1,r2: pwave(r1,r2,df,**kwargs)
    elif mode=="pwave": 
        weightf = lambda r1,r2: pwave(r1,r2,df,**kwargs)
    elif mode=="nodal_fwave":
        #weightf = lambda r1,r2: nodal_fwave(r1,r2,df,H=self,**kwargs)
        weightf = nodal_fwave_generator(df,H=self,**kwargs)
#    elif mode=="chiral_fwave": 
#        weightf = lambda r1,r2: get_triplet(r1,r2,df,L=3)
    elif mode=="chiral_pwave": 
        weightf = lambda r1,r2: get_triplet(r1,r2,df,L=1,**kwargs)
    elif mode=="chiral_fwave": 
        weightf = get_triplet_generator(df,L=3,H=self,**kwargs)
    elif mode=="chiral_dwave": 
        weightf = lambda r1,r2: get_singlet(r1,r2,L=2,**kwargs)
    elif mode=="chiral_gwave": 
        weightf = lambda r1,r2: get_singlet(r1,r2,L=4,**kwargs)
    elif mode=="antihaldane":
        f = get_haldane_function(self.geometry,stagger=True)
        weightf = lambda r1,r2: f(r1,r2)*np.identity(2)
    elif mode=="haldane":
        f = get_haldane_function(self.geometry,stagger=False)
        weightf = lambda r1,r2: f(r1,r2)*np.identity(2)
    elif mode=="swavez":
        weightf = lambda r1,r2: same_site(r1,r2)*tauz
    elif mode=="px":
        weightf = lambda r1,r2: px(r1,r2)
    elif mode=="dpid":
        weightf = lambda r1,r2: dpid(r1,r2,**kwargs)
    elif mode=="swaveA":
        weightf = lambda r1,r2: swaveA(self.geometry,r1,r2)
    elif mode=="swaveB":
        weightf = lambda r1,r2: swaveB(self.geometry,r1,r2)
    elif mode=="swavesublattice":
        def weightf(r1,r2):
          return swaveB(self.geometry,r1,r2) - swaveA(self.geometry,r1,r2)
    elif mode in ["dx2y2","nodal_dwave"]:
        weightf = lambda r1,r2: dx2y2(r1,r2,H=self,**kwargs)
    elif mode=="dxy":
        weightf = lambda r1,r2: dxy(r1,r2,H=self,**kwargs)
    elif mode=="snn":
        weightf = lambda r1,r2: swavenn(r1,r2)
    elif mode=="C3nn":
        weightf = lambda r1,r2: C3nn(r1,r2)
    elif mode=="SnnAB":
        weightf = lambda r1,r2: SnnAB(self.geometry,r1,r2)
    else: raise
    matrixf = lambda r1,r2: deltaf((r1+r2)/2.)*weightf(r1,r2) 
    return matrixf # return function

# matrices for the e-h subsector
iden = np.array([[1.,0.],[0.,1.]],dtype=np.complex)
tauz = np.array([[1.,0.],[0.,-1.]],dtype=np.complex)
taux = np.array([[0.,1.],[1.,0.]],dtype=np.complex)
tauy = np.array([[0.,1j],[-1j,0.]],dtype=np.complex)
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



def C3nn(r1,r2):
    """Function with first neighbor C3 profile"""
    dr = r1-r2
    dr2 = dr.dot(dr)
    if 0.99<dr2<1.001: # first neighbor
#        return dr[0]
        phi = np.arctan2(dr[1],dr[0]) # angle
        return 1.0*np.exp(1j*phi)*tauz
    return 0.0*tauz



def SnnAB(g,r1,r2):
    """Swave between AB"""
    dr = r1-r2
    dr2 = dr.dot(dr)
    if 0.99<dr2<1.001: # first neighbor
        i = g.get_index(r1,replicas=True)
        j = g.get_index(r2,replicas=True)
        if g.sublattice[i]==1 and g.sublattice[j]==-1:
          return 1.0*tauz
        else: return -1.0*tauz
#          return 1.0*np.matrix([[1.,0.],[0.,0.]])
    return 0.0*iden




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
    if abs(L)%2!=1: raise
    if np.abs(dr2-dist2)<1e-4:
        phi = np.arctan2(dr[1],dr[0])
        d = df((r1+r2)/2.) # evaluate dvector
        delta = dvector2delta(d) # compute the local deltas
        ms = np.array([[delta[2],delta[0]],[delta[1],-delta[2]]])
        return np.exp(1j*phi*L)*np.array(ms,dtype=np.complex)
    else: return 0.0*tauz


def get_singlet(r1,r2,L=2,phi0=0.,H=None,nn=1):
    """Function for p-wave order"""
    if L%2!=0: raise
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
