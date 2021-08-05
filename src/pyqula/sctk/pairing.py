
from ..geometry import same_site
import numpy as np
from ..kanemele import get_haldane_function
from ..utilities import get_callable
from .dvector import dvector2delta


def pairing_generator(self,delta=0.0,mode="swave",d=[0.,0.,1.]):
    """Create a geenrator, taking as input two positions, and returning
    the 2x2 pairing matrix"""
    # wrapper for the amplitude and d-vector
    deltaf = get_callable(delta) # callable for the amplitude
    df = get_callable(d) # callable for the d-vector
    if callable(mode):
        weightf = mode # mode is a function returning a 2x2 pairing matrix
    elif mode=="swave":
        weightf = lambda r1,r2: same_site(r1,r2)*np.identity(2)
    elif mode=="triplet": 
        weightf = lambda r1,r2: pwave(r1,r2,df)
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
    elif mode=="swaveA":
        weightf = lambda r1,r2: swaveA(self.geometry,r1,r2)
    elif mode=="swaveB":
        weightf = lambda r1,r2: swaveB(self.geometry,r1,r2)
    elif mode=="swavesublattice":
        def weightf(r1,r2):
          return swaveB(self.geometry,r1,r2) - swaveA(self.geometry,r1,r2)
    elif mode=="dx2y2":
        weightf = lambda r1,r2: dx2y2(r1,r2)
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


def dx2y2(r1,r2):
    """Function with first neighbor dx2y2 profile"""
    dr = r1-r2
    dr2 = dr.dot(dr)
    if 0.99<dr2<1.001: # first neighbor
        return (dr[0]**2 - dr[1]**2)*iden
    return 0.0*iden


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


def pwave(r1,r2,df):
    """Function with first neighbor px profile"""
    dr = r1-r2 ; dr2 = dr.dot(dr)
    if 0.99<dr2<1.001: 
        d = df((r1+r2)/2.)
        delta = dvector2delta(d) # compute the local deltas
        ms = np.array([[delta[2],delta[0]],[delta[1],-delta[2]]])
        return (dr[0]+1j*dr[1])*ms
    return 0.0*tauz

