import numpy as np

def multireplicas(self,n):
    """
    Return several replicas of the unit cell, it is similar
    to supercell but without the shift of the center and going to positive
    and negative positions
    """
    if n==0: return self.r # return positions
    out = [] # empty list with positions
    dl = self.neighbor_directions(n) # list with neighboring cells to take
    if self.dimensionality==0: return self.r
    else: # bigger dimensionality
        for d in dl:  
            out += self.replicas(d=d).tolist() # add this direction
    return np.array(out)

def replicas(self,d=[0.,0.,0.]):
    """Return replicas of the atoms in the unit cell"""
    return np.array([ri + self.a1*d[0] + self.a2*d[1] + self.a3*d[2] for ri in self.r])


def replicate_site(self,ri,n=2):
    """Make replicas of a single site"""
    if n==0: return [self.ri] # return positions
    out = [] # empty list with positions
    dl = self.neighbor_directions(n) # list with neighboring cells to take
    if self.dimensionality==0: return [self.ri]
    else: # bigger dimensionality
        out = np.array([ri + self.a1*d[0] + self.a2*d[1] + self.a3*d[2] for d in dl])
    return np.array(out) # return positions



def get_distance2closest(r,g=None):
    """Distance to the closest site, including replicas"""
    if g is None: rs = [r]
    else: rs = replicate_site(g,r) # return all the replicas
    def fdis(r,r0): # function to compute the distance
        do = np.zeros(len(rs)) # initialize
        for i in range(len(rs)):
            dr = rs[i] - r0 # distance
            do[i] = np.sqrt(dr.dot(dr)) # store
        return np.min(do) # minimum
    return fdis # return function







