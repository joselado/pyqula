import numpy as np
from .. import algebra


def berry_green_generator(f,k=[0.,0.,0.],dk=1e-4,operator=None,
              full=False,gI=None):
  """Function that returns the energy resolved Berry curvature.

  gI, if given, is a callable gI(e=e,k=k) returning the *exact* inverse
  Green's function at the central k-point (e.g. (e+i*delta)*I - H(k),
  already available for free from get_gk_gen's inv=True branch). When
  omitted, gI is approximated by averaging the four finite-difference
  Green's functions and inverting that average -- one extra O(N^3)
  inversion per energy point, and only an approximation of the exact
  central value."""
  k = np.array(k) # set as array
  dx = np.array([dk,0.,0.])
  dy = np.array([0.,dk,0.])
  def fint(e): # function to integrate
#    g = f(e=e,k=k) # compute at this k and this energy
    gxp = f(e=e,k=k+dx) # compute at this k and this energy
    gxm = f(e=e,k=k-dx) # compute at this k and this energy
    gyp = f(e=e,k=k+dy) # compute at this k and this energy
    gym = f(e=e,k=k-dy) # compute at this k and this energy
    if gI is not None: gI0 = gI(e=e,k=k) # exact inverse, no extra inversion
    else:
        g = (gxp + gyp + gxm + gym)/4. # average Green function
        gI0 = algebra.inv(g) # inverse
    # the derivative of g^-1 is -g^-1*g'*g^-1
    omega = -(gxp-gxm)@gI0@(gyp-gym) + (gyp-gym)@gI0@(gxp-gxm)
#    omega = ((gxp-gxm)@(gyp-gym) - (gyp-gym)@(gxp-gxm))@gI
#    omega = g*((gxp.I-gxm.I)*(gyp-gym) -(gyp.I-gym.I)*(gxp-gxm))
#    omega += -g*(gyp.I-gym.I)*(gxp-gxm)
    if operator is not None: omega = operator(omega,k=k)
    if full: return omega/(4.*dk*dk*2.*np.pi) # return the full matrix
    else: return np.trace(omega)/(4.*dk*dk*2.*np.pi) # return contribution
  return fint # return the function



def berry_green(f,emin=-10.0,k=[0.,0.,0.],ne=100,dk=1e-4,operator=None,gI=None):
  """Return the Berry curvature using Green functions. This function
  integrated the Green's function expression using complex integration"""
  import scipy.integrate as integrate
  fint = berry_green_generator(f,k=k,dk=dk,operator=operator,gI=gI)
  es = np.linspace(emin,0.,ne) # energies used for the integration
  ### The original function is defined in the coplex plane,
  # we will do a change of variables of the form z = re^(iphi) - r0
  # so that dz = re^(iphi) i dphi
  def fint2(x):
    """Function to integrate using a complex contour, from 0 to 1"""
    z0 = emin*np.exp(-1j*x*np.pi)/2.
    z = z0 + emin/2.
    return -(fint(z)*z0).imag*np.pi # integral after the change of variables
  return integrate.quad(fint2,0.0,1.0,limit=60,epsabs=0.1,epsrel=0.1)[0]
#  return integrate.quad(fint,emin,0.0,limit=60,epsabs=0.01,epsrel=0.01)[0]
#  return np.sum([fint(e) for e in es]) # return




def berry_operator(h,delta=1e-1,**kwargs):
    """Return ap operator that computes the Berry curvature for a certain
    wavefunction"""
    if h.dimensionality!=2:
        raise ValueError("the Berry operator is only defined for 2d "
                "Hamiltonians")
    h = h.copy()
    hk = h.get_hk_gen() # get generator
    gk = h.get_gk_gen(delta=delta) # get generator
    # The Green mode has some numerical instability,
    # this should be further checked later
    if not h.is_sparse: # dense Hamiltonians
        def gI(e=0.,k=[0.,0.,0.]): # exact inverse Green's function, no extra inversion
            return gk(e=e,k=k,inv=True)
        def bk(k):
            return berry_green_generator(gk,k=k,dk=delta/100,
                    full=True,gI=gI,**kwargs)
        def outf(w,k=[0.,0.,0.]):
            m = hk(k) # bloch Hamiltonian
            e = algebra.braket_wAw(w,m) # energy
            Be = bk(k)(e)
            Be = (Be + algebra.dagger(Be))/2. # Hermitian
            o = Be@(delta*w) # Berry curvature
            return o # return a vector
        return outf
    else:
        def gI(**kwargs): # workaround for inverse of Green's function
            return gk(**kwargs,inv=True) # make the inverse
        def bk(k): return berry_green_generator_sparse(gk,k=k,gI=gI,**kwargs)
        def outf(w,k=[0.,0.,0.]):
            m = hk(k) # bloch Hamiltonian
            e = algebra.dot(w,m*w) # energy
            w2 = bk(k)(e)@(delta*w) # Operator times WF
            # to think, perhaps this should be <v|B|v>|v>
            # currently it is B|v>
            # (this does not change the expectation value though)
            return w2 # return a vector
        return outf




def berry_green_generator_sparse(f,k=[0.,0.,0.],
        dk=1e-4,gI=None):
  """Function that returns the energy resolved Berry curvature"""
  k = np.array(k) # set as array
  dx = np.array([dk,0.,0.])
  dy = np.array([0.,dk,0.])
  def fint(e): # function to return
    gxp = f(e=e,k=k+dx) # compute at this k and this energy
    gxm = f(e=e,k=k-dx) # compute at this k and this energy
    gyp = f(e=e,k=k+dy) # compute at this k and this energy
    gym = f(e=e,k=k-dy) # compute at this k and this energy
    # Now apply the formula
    gI0 = gI(e=e,k=k) # inverse of the Green function
    # the derivative of g^-1 is -g^-1*g'*g^-1
    omega = -(gxp-gxm)@gI0@(gyp-gym) + (gyp-gym)@gI0@(gxp-gxm)
    return omega/(4.*dk*dk*2.*np.pi) # return the full matrix
  return fint # return the function




def default_dk(delta):
  """Finite-difference step in k for a Green's function broadened by delta"""
  return delta/25.


def dOmega_dE_generator(h,operator=None,delta=0.02,dk=None):
  """Return a function that generates
  dOmega/dE (k,E), the derivative of the Berry curvature
  with respect to the chamical potential.

  dk is the finite-difference step in k, in reduced coordinates. It must
  be small against delta, since near a band edge the bands move by a few
  times dk in units of t, so the default is delta/25; a fixed step of 0.02
  underestimated the plateau of chern_density by 0.07 at delta=0.05
  (future_development/guide_figures_followups.md)"""
  if dk is None: dk = default_dk(delta)
  gf = h.get_gk_gen(delta=delta,canonical_phase=True) # green function generator
  def gI(e=0.,k=[0.,0.,0.]): # exact inverse Green's function, no extra inversion
      return gf(e=e,k=k,inv=True)
  # Return dOmega/dE
  def f(k=[0.,0.,0.],e=0.):
      fgreen = berry_green_generator(gf,k=k,dk=dk,operator=operator,full=False,gI=gI)
      # -1: berry_green_generator's raw fint(e), evaluated directly on the
      # real energy axis, comes out with the opposite sign of the
      # independently Wilson-loop-validated complex-contour integration in
      # berry_green (which consumes the very same fint via a change of
      # variables) -- verified at a single k-point, where the real-axis
      # integral of dOmega/dE from deep in the band to e=0 reproduces
      # -berry_green(...) to 5 significant figures. Without this sign flip,
      # chern_density's cumulative energy integral converges to minus the
      # Chern number instead of matching it.
      return -fgreen(e).real
  return f


def dOmega_dE_vertical(h,x,ks,operator=None,delta=0.02,dk=None,nodes=8):
  """Sum over the k-points ks of the integral of the analytic continuation
  of dOmega/dE along the vertical segment from x to x+i*delta, with a
  Gauss-Legendre rule of nodes points in the imaginary part y (no node at
  y=0). The cumulative integral of chern_density runs along the line
  Im z = delta; adding this segment at its two ends moves the path down to
  the real axis, which removes the broadening exactly, as long as x is
  below the spectrum or inside a gap, where the integrand is smooth down
  to y=0. Returns the complex sum, with the normalization of
  berry_green_generator (not yet divided by the number of k-points)"""
  if dk is None: dk = default_dk(delta)
  ys,ws = np.polynomial.legendre.leggauss(nodes)
  ys = (ys+1.)*delta/2. ; ws = ws*delta/2. # from [-1,1] to [0,delta]
  out = 0.
  for (y,w) in zip(ys,ws): # loop over the imaginary parts
      gf = h.get_gk_gen(delta=y,canonical_phase=True) # G(x+iy)
      def gI(e=0.,k=[0.,0.,0.]): return gf(e=e,k=k,inv=True)
      for k in ks:
          f = berry_green_generator(gf,k=k,dk=dk,operator=operator,
                  full=False,gI=gI)
          out += w*f(x)
  return out


def dOmega_dE(h,k=[0.,0.,0.],e=0.,**kwargs):
  """Compute dOmega/dE (k,E), the derivative of the Berry curvature
  with respect to the chamical potential"""
  return dOmega_dE_generator(h,**kwargs)(k=k,e=e)





def berry_green_rmap_kpoint(h,emin=None,k=[0.,0.,0.],
        ne=100,dk=0.0001,operator=None,integral_mode="complex",
                  delta=0.002,integral=True,eps=1e-3,
                  energy=0.0,emax=0.0):
    """Return the spatially resolved Berry curvature at a certain kpoint,
    one value per orbital, either integrated up to the Fermi energy
    (integral=True) or at a single energy (integral=False)"""
    from ..integration import integrate_matrix
    f = h.get_gk_gen(delta=delta,canonical_phase=True) # green function generator
    fgreen = berry_green_generator(f,k=k,dk=dk,operator=operator,full=True)
    # No minimum energy provided
    if emin is None and integral:
        emin = algebra.eigvalsh(h.get_hk_gen()(k))[0] - 1.0
        print("Minimum energy",emin)
    def fint(x):
      return np.diag(fgreen(x)) # return diagonal
    ### The original function is defined in the complex plane,
    # we will do a change of variables of the form z = re^(iphi) - r0
    # so that dz = re^(iphi) i dphi
    if integral: # integrate up to the fermi energy
      def fint2(x):
        """Function to integrate using a complex contour, from 0 to 1"""
        de = emax-emin # energy window of the integration
        ce = de/2. # center of the circle
        z0 = -ce*np.exp(-1j*x*np.pi) # parametrize the circle
        z = z0 + (emin+emax)/2. # shift the circle
        print("Evaluating",x)
        return 1j*(fint(z)*z0)*np.pi # integral after the change of variables
      out = integrate_matrix(fint2,xlim=[0.,1.],eps=eps)
      return (1j*out).imag # Berry curvature
    else: # evaluate at the fermi energy
      return (1j*fint(energy)).imag # Berry curvature
