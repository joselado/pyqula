import numpy as np

def target_moduli(v1,v2,m):
    """Given two vectors, obtain a linear combination with a certain moduli"""
    n = 2*int(m) # estimate of the number
    n = 1
    for i in range(-n,n+1):
        for j in range(-n,n+1):
            v = i*v1 + j*v2
            if abs(np.sqrt(v.dot(v))-m)<1e-3: return v
    raise # not found

def closest_kreplica(g,v,v0):
    """Return the closest replica of a kpoint close to another one"""
    if g.dimensionality==0: return v*0. # for 0d
    elif g.dimensionality==1: # for 1d
        w = v[0]*g.b1 # redefine
        w0 = v0[0]*g.b1 # redefine
        n = 2
        dmax = 1e8
        for i in range(-n,n+1):
                wt = w + g.b1*i # compute replica
                dw = wt - w0 # distance to reference point
                dw = np.sqrt(dw.dot(dw)) # distance
                if 1e-4<dw<dmax: 
                    vo = v + np.array([i,0.,0.])
                    dmax = dw
        return vo # return vo
    elif g.dimensionality==2: # for 2d
        w = v[0]*g.b1 + v[1]*g.b2 # redefine
        w0 = v0[0]*g.b1 + v0[1]*g.b2 # redefine
        n = 2
        dmax = 1e8
        for i in range(-n,n+1):
            for j in range(-n,n+1):
                wt = w + g.b1*i + g.b2*j # compute replica
                dw = wt - w0 # distance to reference point
                dw = np.sqrt(dw.dot(dw)) # distance
                if 1e-4<dw<dmax: 
                    vo = v + np.array([i,j,0.])
                    dmax = dw
        return vo # return vo
    else: raise




def same_kpoint(k1,k2,tol=1e-6):
    """Check if two kpoints are the same"""
    dk = np.array(k1) - np.array(k2) # difference
    dk = np.abs(dk) # absolute value
    dk = np.exp(1j*dk*np.pi*2.) # to the unit circle
    if np.max(np.abs(dk-1.))>tol: return False
    else: return True






def closest_path(g,kps):
    """Given a list of kpoints, return the closest path"""
    out = [kps[0]] # first point
    for i in range(len(kps)-1): # loop over kpoints
        k = closest_kreplica(g,kps[i+1],out[i])
        out.append(k) # store
    return out



def k2path(g,kp,nk=100):
    """Given a set of kpoints, return the path"""
    kp = closest_path(g,kp) # reduced path
    kp = [np.array(k) for k in kp] # convert to arrays
    ks = [] # empty list of kpoints
    for i in range(len(kp)-1): # loop over pairs
      dk0 = kp[i+1] - kp[i] # difference
      dk = dk0[0]*g.b1 + dk0[1]*g.b2 # real reciprocal vector
      dk2 = dk.dot(dk) # moduli real vector
      nk2 = int(nk*np.sqrt(dk2/g.b1.dot(g.b1))) # in units of reciprocal vector
      steps = np.linspace(0.,1.,nk2,endpoint=False) # number of points
      for s in steps:
        ks += [kp[i] + dk0*s] # add kpoint
    ks += [kp[len(kp)-1]] # add the last one
    return ks

