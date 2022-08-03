import numpy as np
from scipy.sparse import csc_matrix


def hartree(h,v=0.0,**kwargs):
    """Add a crystal field to a Hamiltonian"""
    if v==0.0: return
    m = cf_potential(h.geometry,vc=v,**kwargs) # get array
    m = m - np.min(m) # shift to zero
    if np.max(np.abs(m))>1e-7: m = m/np.max(np.abs(m)) # normalize
    m = m - np.mean(m) # remove average
    nat = len(h.geometry.r) # number of atoms
    ind = range(nat) # indexes
    mat = v*csc_matrix((m,(ind,ind)),shape=(nat,nat),dtype=np.complex)
    h.intra = h.intra + h.spinless2full(mat) # add contribution


def hartree_onsite(g,**kwargs):
    return cf_potential(g,**kwargs,mode="full")



def cf_potential(g,rcut=6.0,vc=0.0,mode="full"):
    """Return an array with the Hartree terms"""
# create fucntions to compute effective distance
    if mode=="full": # full mode
        def getd(dr,dx,dy,dz): return dr
    elif mode=="stacking": # stacking mode
        def getd(dr,dx,dy,dz):
            dr = dx**2 + dy**2 + 0.1*dz**2
            return dr
    else: raise # not implemented
    g = g.copy() # copy geometry
    interactions = [] # empty list
    nat = len(g.r) # number of atoms
    mout = np.zeros(nat) # initialize array
    if g.dimensionality>0:
      lat = np.sqrt(g.a1.dot(g.a1)) # size of the unit cell
      g.ncells = int(2*rcut/lat)+2 # number of unit cells to consider
    ri = g.r # positions
    for d in g.neighbor_directions(): # loop
        rj = np.array(g.replicas(d=d)) # positions
        for i in range(nat):
            dx = rj[:,0] - ri[i,0]
            dy = rj[:,1] - ri[i,1]
            dz = rj[:,2] - ri[i,2]
            dr = dx*dx + dy*dy + dz*dz
            dr = np.sqrt(dr) # square root
            dr = getd(dr,dx,dy,dz) # effective distance
            dr[dr<1e-4] = 1e10
#            v = vc/dr # Coulomb interaction
            v = vc*np.exp(-dr/rcut) # quench interaction
            #v = v*1./(dr/rcut) # quench interaction
            v[dr<1e-4] = 0.0
            v[dr>rcut] = 0.0
            mout[i] += np.sum(v) # store contribution
    return mout # return



def bulkedge_function(g,vbulk=0.0,vedge=1.0,rcut=20.0,sharpness=None,
        ebcut=.5):
    """Return a function that distinguishes between bulk and edge"""
    m = hartree_onsite(g,vc=1.0,rcut=rcut) # compute the array
    m = m - np.min(m) # shift
    m = m/np.max(m) # normalize
    if not sharpness is None:  
        m = (np.tanh(sharpness*(m-ebcut)) + 1.)/2.
    m = vedge + (vbulk-vedge)*m # scale
    from .geometry import array2function
    f = array2function(g,m) 
    return f # return function




