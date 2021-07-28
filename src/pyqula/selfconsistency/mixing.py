import numpy as np

class Mixing():
    def __init__(self,n=6):
        """Create the mixing object"""
        self.n = n
        self.errors = np.zeros(n) # empty array
        self.full = False
        self.i = 0 # number of stored
    def get_mix(self,d):
        """Get the new mixing"""
        self.errors = np.concatenate([self.errors[1:self.n],[d]]) # store
        if self.full: # filled mixing
            m = self.errors[self.n-1]/self.errors[self.n-2] # decrease
            if abs(m-1.0)<1e-2: return 0.01 # slow mixing
            elif m>1.: return 0.5 # increasing, normal mixing
            else: return 1.-m
        else:
            self.i += 1 # increase counter
            if self.i == (self.n-1): self.full = True
            return 0.5

