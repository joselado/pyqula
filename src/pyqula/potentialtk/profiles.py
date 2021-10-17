import numpy as np

def radial_decay(v0=2.0,rl=3.0,voo=1.0,mode="exp"):
    """Create a fucntion that implements a radial decay
    at a certain distance"""
    if mode=="exp":
        def fs(r):
            r2 = r.dot(r)
            return voo + (v0-voo)*np.exp(-(1./rl)*np.sqrt(r2))
        return fs
    elif mode=="linear":
        def fs(r):
            r0 = np.sqrt(r.dot(r))
            if r0>rl: return voo
            else: return v0 + (voo-v0)*r0/rl
        return fs
    else: raise


