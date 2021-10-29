import numpy as np

def randomwf(nd):
    """Return a random wavefunction of simension n"""
    def fun():
      v = (np.random.random(nd) - 0.5)*np.exp(2*1j*np.pi*np.random.random(nd))
      return v/np.sqrt(np.sum(np.abs(v)**2)) # normalize
    return fun
