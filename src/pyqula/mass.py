import numpy as np

# functions to compute the effective mass


def mass_operator(h,**kwargs):
    """Return the operator to compute the effective mass"""
    def mop(v,k=None):
        if k is None: 
            print("Kpoint required")
            raise
        hk0 = h.get_hk_gen()(k) # get the Hamiltonian
        from .algebra import eigh # function to diagonalize
        e0,w0 = eigh(hk0) # central point 
        m = effective_mass_velocity(h,k,w0=w0,**kwargs) # array with masses
        T = np.matrix(w0) # convert
        m0 = np.diag(m) # build new hamiltonian
        #O = np.array(np.conjugate(T)@m0@T.T) # operator
        #O = np.conjugate(np.array(T@m0@T.H)) # operator
        O = np.array(T@m0@T.H) # operator
        out =  O@v # return the result
        return out
    from .operators import Operator
    return Operator(mop)


def effective_mass(h,k,dk=1e-2,**kwargs):
    """Given a Hamiltonian and a k-point, return the effective mass"""
    from .topology import smooth_gauge # perform rotation if needed
    from .algebra import eigvalsh # function to diagonalize
    if h.dimensionality==1: # not implemented
        k = np.array(k)
        hk = h.get_hk_gen() # get generator
        dkv = np.array([dk,0.,0.])
        e0 = np.sort(eigvalsh(hk(k)))
        ep = np.sort(eigvalsh(hk(k+dk)))
        em = np.sort(eigvalsh(hk(k-dk)))
        m = ((e0 - em) - (ep - e0))/(dk*dk) # array with effective mass
        a1 = h.geometry.a1
        v = np.sqrt(a1.dot(a1))
        m = v*m/((2.*np.pi)**2) # renormalize by the pi factors
        return m
    else: raise




def effective_mass_velocity(h,k,dk=1e-3,w0=None):
    """Given a Hamiltonian and a k-point, return the effective mass"""
    from .topology import smooth_gauge # perform rotation if needed
    from .algebra import eigh # function to diagonalize
    from .algebra import braket_wAw # function to diagonalize
    if h.dimensionality==1: # not implemented
        k = np.array(k)
        hk = h.get_hk_gen() # get generator
        dkv = np.array([dk,0.,0.])
        hk0 = hk(k)
#        hkp = hk(k+dkv)
#        hkm = hk(k-dkv)
        if w0 is None: # initial waves not provided
            e0,w0 = eigh(hk0) # central point 
            w0 = w0.T.copy() # transpose
        else:
            w0 = w0.T.copy() # transpose
            e0 = np.array([braket_wAw(w,hk0) for w in w0]) # w0 is given
#        em,wm = eigh(hkm) # previous wavefunctions
#        ep,wp = eigh(hkp) # next wavefunctions
#        wm = wm.T
#        wp = wp.T
        from .current import current_operator
        J = current_operator(h) # current generator
        from .current import derivative
        mass = lambda k: derivative(h,k,order=[2])
        vm = np.array([braket_wAw(w,J(k-dkv)) for w in w0]) # velocity before
        vp = np.array([braket_wAw(w,J(k+dkv)) for w in w0]) # velocity before
#        wm = smooth_gauge(w0,wm).T # perform smooth rotation
#        wp = smooth_gauge(w0,wp).T # perform smooth rotation
#        em = np.array([braket_wAw(w,hkm) for w in wm]) # energies before
#        ep = np.array([braket_wAw(w,hkp) for w in wp]) # energies after
#        m = ((e0 - em) - (ep - e0))/(dk*dk) # array with effective mass
        m = (vp-vm)/dk
        m = np.array([braket_wAw(w,mass(k)) for w in w0]) # velocity before
#        print(e0.real)
#        return (e0 - em)/(dk) # velocity
        a1 = h.geometry.a1
        v = np.sqrt(a1.dot(a1))
        m = v*m/((2.*np.pi)**2) # renormalize by the pi factors
        return m # return the array with masses in the diagonal basis
    else: raise


