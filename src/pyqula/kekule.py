import numpy as np
from . import geometry


remove_duplicated = geometry.remove_duplicated_positions

### Routines dealing with Kekule order ###

def kekule_positions(r):
    """
    Returns the positions defining a Kekule ordering
    """
    cs = hexagon_centers(r,r) # return the centers
    cs = remove_duplicated(cs) # remove duplicated
    cs = retain(cs,d=3.0) # retain only centers that are at distance 2
    return np.array(cs) # return array



def kekule_center(r):
    """
    Returns a function that given two positions, returns the Kekule center
    """
    cs = kekule_positions(r) # returns centers of the kekule
    def f(r1,r2): # function that returs the kekule center
        dr = r1-r2 ; dr=dr.dot(dr) 
        if not 0.99<dr<1.01: return None # too far
        for c in cs: # loop over centers
            # ensure that they are in the "hexagon"
            dr1 = c-r1 ; dr1 = dr1.dot(dr1) # distance to r1
            dr2 = c-r2 ; dr2 = dr2.dot(dr2) # distance to r2
            if 0.99<dr1<1.01 and 0.99<dr2<1.01: # if clse to center
              return c # return the center
        return None # no center found
    return f # return function






def kekule_function(r,t=1.):
    """
    Returns a function that will compute Kekule hoppings
    """
    cs = kekule_positions(r) # get the centers with all the positions
    ## Define a function to only have hoppings in the hexagon
    if callable(t): fun = t # t is a function
    else: # assume t is a number
        def fun(r1,r2):
            dr = r1-r2 ; dr = dr.dot(dr) # distance 1
            if 0.99<dr<1.01: return t # first neighbor
            else: return 0.0 # nothing
    def f(r1,r2):
        for c in cs: # loop over centers
            dr = r1-r2 ; dr=dr.dot(dr) 
            if not 0.99<dr<1.01: continue
            # ensure that they are in the "hexagon"
            dr1 = c-r1 ; dr1 = dr1.dot(dr1) # distance to r1
            dr2 = c-r2 ; dr2 = dr2.dot(dr2) # distance to r2
            if 0.99<dr1<1.01 and 0.99<dr2<1.01: # if clse to center
              return fun(r1,r2) # return the hopping
        return 0.0 # no hopping
    # now define the function
    def fm(rs1,rs2):
      m = np.zeros((len(rs1),len(rs2)),dtype=np.complex) # initialize matrix
      for i in range(len(rs1)): # loop
        for j in range(len(rs2)): # loop
            m[i,j] = f(rs1[i],rs2[j]) # get kekule coupling
      return m # return the Kekule matrix
    return fm # return the function


def kekule_matrix(r1,r2=None,**kwargs):
    """
    Return a Kekule matrix for positions r, assuming
    they are from a honeycomb-like lattice
    """
    if r2 is None: r2 = r1
    f = kekule_function(r1,**kwargs)
    return f(r1,r2)


def hexagon_centers(r1,r2):
    """
    Return the centers of an hexagon
    """
    out = []
    for ri in r1: # loop
        for rj in r2: # loop
            dr = ri-rj
            dr = dr.dot(dr) # distance
            if 3.9<dr<4.1: # center of an hexagon
                out.append((ri+rj)/2.) # store the center
    return out # return list with centers


def r_in_rs(r,rs):
    """
    Check that a position is not stored
    """
    for ri in rs:
        dr = ri-r ; dr = dr.dot(dr)
        if dr<0.01: return True
    return False



def retain(r,d=3.0):
    """
    Retain only sites that are at a distance d
    """
    i = np.random.randint(len(r))
    out = [r[0]] # take first one
    def iterate(out): # do one iteration
      out0 = [r for r in out] # initialize
      for rj in out: # loop over stored
        for ri in r: # loop
            dr = ri-rj ; dr = dr.dot(dr) # distance
            if d*d-0.1<dr<d*d+0.1: # if desired distance
                # now check that this one has not been stored already
                if not r_in_rs(ri,out0): # not stored yet
                  out0.append(ri) # store position
      return out0
#    np.savetxt("R.OUT",np.matrix(r))
#    exit()
    while True:
#    for i in range(10):
        out1 = iterate(out) # do one iteration
        out1 = remove_duplicated(out1) # remove duplicated atoms
        if len(out1)==len(out): break
        out = [r for r in out1] # redefine
#    np.savetxt("R.OUT",np.matrix(out)) # write in file
#    exit()
    return out # return desired positions




def chiral_kekule(g,t1=0.0,t2=0.0,hermitian=True):
    """Add a chiral kekule hopping, input is a geometry"""
    fkc = kekule_center(g.r) # function for kekule center
    dr0 = g.r[0] - g.r[1] # one NN vector
    dz = np.exp(1j*2.*np.pi/3.) # second vector
    z0 = dr0[0] + 1j*dr0[1] # complex NN vector
    if not 0.99<dr0.dot(dr0)<1.01: raise # this has to be more robust
    # function to check first kind of bond
    def kind1(dr):
        z = dr[0] + 1j*dr[1] # complex position
        for iz in [z0,z0*dz,z0*dz*dz]: # loop
            if abs(z*np.conjugate(iz)-1.0)<0.01: return True
        return False
    # function to check if going clockwise
    clockwise = lambda r1,r2: np.cross(r1,r2)[2]>0. # if clockwise
    tz = dz # complex hopping
    tz = 1.0
    def fun(r1,r2):
        dr = r1-r2 # distance between sites
        if not 0.99<dr.dot(dr)<1.01: return 0. # not first neighbors
        rk = fkc(r1,r2) # Kekule center
        if clockwise(rk-r1,r2-r1): # clockwise hopping
          if kind1(dr): return t1 # first kind of bond
          else: return t2
        else:
          if hermitian: # conventional Hamiltonian
            if kind1(dr): return np.conjugate(t2) # first kind of bond
            else: return np.conjugate(t1)
          else: return 0.0
    return fun 





