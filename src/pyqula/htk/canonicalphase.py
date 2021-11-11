import numpy as np

def canonical_unitary(self,fin):
    """Return a function that wraps any matrix inot ints canonical unitary"""
    def fun(k):
        frac_r = self.geometry.frac_r # fractional coordinates
        frac_r = frac_r - frac_r[0]
        fphase = lambda ri: np.exp(1j*np.pi*2.*ri.dot(k)) # phase
        U = np.diag([fphase(r) for r in frac_r])
        U = np.array(U) # this is without .H
        U = self.spinless2full(U) # increase the space if necessary
        hk = fin(k)
        Ud = np.conjugate(U.T) # dagger
        hk = Ud@hk@U
        return hk # return matrix
    return fun



def canonical_unitary_generator(self):
    """Return a function that wraps any matrix inot ints canonical unitary"""
    raise
    def fun(k=None,d=None):
        """Given a certain directional hopping, return the Bloch matrix"""
        frac_r = self.geometry.frac_r # fractional coordinates
        frac_r = frac_r - frac_r[0] + np.array(d) # new fractional coordinates
        fphase = lambda ri: np.exp(1j*np.pi*2.*ri.dot(k)) # phase
        U = np.diag([fphase(r) for r in frac_r])
        U = np.array(U) # this is without .H
        U = self.spinless2full(U) # increase the space if necessary
        return U
    return fun # return the generating function

def canonical_unitary_hopping(self):
    """Return the generator for the Bloch hoppings"""
    fb = canonical_unitary_generator(self) # generating unitaries
    def fun(m,d=[0,0,0],**kwargs):
        U1 = fb(d=[0,0,0],**kwargs) # get unitary
        U2 = fb(d=d,**kwargs) # get unitary
        U1d = np.conjugate(U1).T
        U2d = np.conjugate(U2).T
        return U1d@m@U2 # return unitary
    return fun
