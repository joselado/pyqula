import numpy as np

def get_local_energy(CS):
    """Return the local energy at each site for the current snapshot.

    A bond (i,j) between two different sites is shared, half of it to each
    site, so that the local energies add up to the total energy. A pair
    (i,i), a site coupled to its own periodic images, belongs to site i
    alone and is counted fully, as in the Ising and lattice-gas models"""
    def get(ii): # get for site ii
        CS0 = CS.copy() # make a dummy copy
        pairs0 = [] # empty list
        j0 = [] # empty list
        for (p,j) in zip(CS.pairs,CS.j): # take only those for this site
            if ii==p[0] or ii==p[1]: # site ii is here
                pairs0.append(p)
                if p[0]==p[1]: j0.append(j) # the site with its own images
                else: j0.append(j/2.) # half of a bond to another site
        b0 = CS.b*0. # zeros
        b0[ii] = CS.b[ii]
        if len(pairs0)==0: # site has no exchange interactions
            CS0.pairs = np.zeros((0,2),dtype=int)
            CS0.j = np.zeros((0,3,3))
        else:
            CS0.pairs = np.array(pairs0)
            CS0.j = np.array(j0)
        CS0.b = b0
        enii = CS0.get_energy()
        return enii
    return np.array([get(ii) for ii in range(len(CS.geometry.r))]) # loop over positions
