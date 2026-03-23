from scipy.signal import hilbert
from .. import algebra
import numpy as np
from .green import berry_green_generator
from ..integration import integrate_matrix

#def get_QG_dE(h,k=[0.,0.,0.],delta=1e-1,**kwargs):
#    """Compute the dqg/dE (k,E), the derivative of the 
#    quantum geometry with respect to the energy
#    by using a Hilbert transform"""
#    es = np.linspace(-6,6,int(50/delta)) # energies
#    from .green import dOmega_dE_generator
#    bef = dOmega_dE_generator(h,delta=delta,**kwargs) # generator of the Berry curvature
#    bes = np.array([bef(k=k,e=e) for e in es]) # Berry curvature at the different energies
#    yz = hilbert(bes) # Hilbert transform
#    qge = np.imag(yz) # pick the imaginary component
#    return es,qge,bes # return energy, dQG/dE and dOmega/dE
#
#
#def get_QG(h,delta=1e-1,**kwargs):
#    """Compute the quantum geometry (and Berry curvature) by summing all the
#    accupied states"""
#    es,qge,bes = get_QG_dE(h,delta=delta,**kwargs) # return dQG/dE and dOmega/dE
#    dE = es[1]-es[0] # dE
#    weight1 = (1.-np.tanh(es/delta))/2. # weight function (soft step)
#    weight2 = delta/(es**2 + delta**2)*1./np.pi # weight function (soft delta)
#    qgew = qge*weight2 # at fermi with a smearing
#    besw = bes*weight1 # below fermi with a smearing
#    return np.trapz(qgew,dx=dE), np.trapz(besw,dx=dE) # return the integral


#def get_QG_kpath(h,kpath=None,nk=100,**kwargs):
#    from ..import klist
#    kpath = klist.get_kpath(h.geometry,kpath=kpath,nk=nk) # take default kpath
#    out = np.array([get_QG(h,k=k,**kwargs) for k in kpath]) # compute all
#    qg = out[:,0] # quantum geometry
#    be = out[:,1] # Berry curvature
#    inds = np.array(range(len(qg))) # index counter
#    return inds,qg,be
    


def QG_green(H,emin=-10.0,k=[0.,0.,0.],ne=100,dk=1e-4,
             delta=1e-3,operator=None):
    """Return the quantum geometry using Green functions. This function
    integrated the Green's function expression using complex integration"""
    import scipy.integrate as integrate
    f = H.get_gk_gen(delta=delta) # get generator
    fint = berry_green_generator(f,k=k,dk=dk,operator=operator)
    es = np.linspace(emin,0.,ne) # energies used for the integration
    ### The original function is defined in the coplex plane,
    # we will do a change of variables of the form z = re^(iphi) - r0
    # so that dz = re^(iphi) i dphi
    def fintz(x):
      """Function to integrate using a complex contour, from 0 to 1"""
      z0 = emin*np.exp(-1j*x*np.pi)/2.
      z = z0 + emin/2.
      return -(fint(z)*z0)*np.pi # integral after the change of variables
    def fintr(x): return fintz(x).real # real part
    def finti(x): return fintz(x).imag # imaginary part
    from ..integration import integrate_matrix
    o = integrate_matrix(fintz,xlim=[0.,1.],eps=1e-4)
    return o # return the full quantum geometry
#    ore = integrate.quad(fintr,0.0,1.0,limit=60,epsabs=0.1,epsrel=0.1)[0]
#    oim = integrate.quad(finti,0.0,1.0,limit=60,epsabs=0.1,epsrel=0.1)[0]
#    o = ore + 1j*oim
#    return o # return the complex number



def get_QG_kpath(h,kpath=None,nk=100,**kwargs):
    from ..import klist
    kpath = klist.get_kpath(h.geometry,kpath=kpath,nk=nk) # take default kpath
    out = np.array([QG_green(h,k=k,**kwargs) for k in kpath]) # compute all
    out = np.array(out) # as array
    qg = out.real # real part
    be = out.imag # imaginary part
    inds = np.array(range(len(qg))) # index counter
    return inds,qg,be




def QG_green_rmap_kpoint(h,emin=None,k=[0.,0.,0.],
        ne=100,dk=0.0001,operator=None,integral_mode="complex",
                  delta=0.002,integral=True,eps=1e-3,
                  energy=0.0,emax=0.0):
    """Return the quantum geometry map at a certain kpoint. This function
    allows to compute both dQG/dE, and QG"""
    f = h.get_gk_gen(delta=delta,canonical_phase=True) # green function generator
    fgreen = berry_green_generator(f,k=k,dk=dk,operator=operator,full=True)
    # No minimum energy provided
    if emin is None and integral:
        emin = algebra.eigvalsh(h.get_hk_gen()(k))[0] - 1.0
        print("Minimum energy",emin)
    def fint(x):
  #    return fgreen(x).trace()[0,0] # return diagonal
      return np.diag(fgreen(x)) # return diagonal
    ### The original function is defined in the complex plane,
    # we will do a change of variables of the form z = re^(iphi) - r0
    # so that dz = re^(iphi) i dphi
    if integral: # integrate up to the fermi energy
      es = np.linspace(emin,0.,ne) # energies used for the integration
      def fint2(x):
        """Function to integrate using a complex contour, from 0 to 1"""
        de = emax-emin # energy window of the integration
        ce = de/2. # center of the circle
        z0 = -ce*np.exp(-1j*x*np.pi) # parametrize the circle
        z = z0 + (emin+emax)/2. # shift the circle
        print("Evaluating",x)
        return 1j*(fint(z)*z0)*np.pi # integral after the change of variables
      out = integrate_matrix(fint2,xlim=[0.,1.],eps=eps)
      return 1j*out # return quantum geometry
    else: # evaluate at the fermi energy
      return 1j*fint(energy) # return quantum geometry





