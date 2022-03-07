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
