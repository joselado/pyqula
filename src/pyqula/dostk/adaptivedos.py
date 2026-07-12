import numpy as np
#from ..algebra import eigvalsh
from scipy.linalg import eigvalsh
from .eigtodos import calculate_dos

# routines to compute DOS using adaptive integration

def adaptive_dos(h,energies=np.linspace(-1.,1.,200),
                 error=1e-1,nk=100,
                 **kwargs):
    """Compute DOS using adaptive integration"""
    if h.is_sparse: raise # only for dense Hamiltonians
    f = generate_function(h,energies=energies,**kwargs) # function to integrate
    dim = h.dimensionality # dimensionality
    limit = max([2,nk//30])
    integrate = get_integrator(error=error,limit=limit) # get the integrator
    workers = 1 # number of workers (this is not yet working)
    if dim==0: # zero dimensional
        return energies,f([0.,0.,0.])
    elif dim==1: # one dimensional 
        out = integrate(lambda kx: f(kx))
        return energies,out
    elif dim==2: # one dimensional 
        def fy(ky):
            def fx(kx):
                return f([kx,ky])
            return integrate(fx,
                             workers=workers # in parallel
                             ) # integrate in x
        out = integrate(fy) # integrate in y
        return energies,out
    else: raise



def get_integrator(error=1e-2,limit=10): # master function for integration
    """Function to perform 1d intergration"""
    from scipy.integrate import quad_vec
    def fint(func,**kwargs):
        epsabs = error # absolute error
        epsrel = error # relative error
        return quad_vec(func,0.,1.,quadrature="gk21",
                       limit=limit,epsabs=epsabs,
                        epsrel=epsrel,**kwargs)[0]
    return fint





def generate_function(h,operator=None,energies=np.linspace(-1.,1.,200),
                 error=1e-1,nk=100,
                 delta=1e-2,**kwargs):
    """Generate function to compute DOS using adaptive integration"""
    if h.is_sparse: raise # only for dense Hamiltonians
    dim = h.dimensionality # dimensionality
    if operator is not None: # use a workaround (not too efficient)
        def f(k):
            if dim==0: k = [0.,0.,0.]
            elif dim==1: k = np.array([k,0.,0.])
            elif dim==2: k = np.array([k[0],k[1],0.])
            else: raise # not implemented
            out = h.get_bands(kpath=[k],operator=operator,
                              write=False,**kwargs)
            w = out[2] # use weight of the bands
            return calculate_dos(out[1],energies,delta,w=w,
                                 parallel=False)
    else:
        hk = h.get_hk_gen() # get Bloch Hamiltonian generator
        def f(k): 
            """Function to integrate"""
            if dim==0: k = [k,0.,0.]
            elif dim==1: k = np.array([k,0.,0.])
            elif dim==2: k = np.array([k[0],k[1],0.])
            else: raise # not implemented
            m = hk(k) # compute Bloch Hamiltonian
            es = eigvalsh(m) # eigenvalues
            return calculate_dos(es,energies,delta,parallel=False)
    return f # return the function


