import numpy as np

def get_local_energy(CS):
    """Return the local energy at each site for the current snapshot"""
    def get(ii): # get for site ii
        CS0 = CS.copy() # make a dummy copy
        pairs0 = [] # empty list
        j0 = [] # empty list
        for (p,j) in zip(CS.pairs,CS.j): # take only those for this site
            if ii==p[0] or ii==p[1]: # site ii is here
                pairs0.append(p)
                j0.append(j)
        b0 = CS.b*0. # zeros
        b0[ii] = CS.b[ii]
        CS0.pairs = np.array(pairs0)
        CS0.j = np.array(j0)/2.
        CS0.b = b0
        enii = CS0.get_energy()
        return enii
    return np.array([get(ii) for ii in range(len(CS.geometry.r))]) # loop over positions

