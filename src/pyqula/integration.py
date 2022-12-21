
import numpy as np


adaptive_error = 1e-2 # error in adaptive algorithm


def integrate_matrix(f,xlim=[0.,1.],eps=0.1,only_imag=False):
  """ Integrates a matrix, the measure is the maximun value of the matrix"""
  return adaptive_simpsons_rule(f,xlim[0],xlim[1],eps,only_imag=only_imag)



def integrate_matrix_2D(f,xlim=[0.,1.],ylim=[0.,1.],**kwargs):
    """Perform a 2D integral of a matrix"""
    def intx(x): # inegrate in the y axis
        return integrate_matrix(lambda y: f([x,y]),xlim=ylim,**kwargs)
    return integrate_matrix(intx,xlim=xlim,**kwargs) # now integrate in x axis
    



def simpsons_rule(f,a,b):
    c = (a+b) / 2.0
    h3 = abs(b-a) / 6.0
    return h3*(f(a) + 4.0*f(c) + f(b))
 
def recursive_asr(f,a,b,eps,whole):
    "Recursive implementation of adaptive Simpson's rule."
    c = (a+b) / 2.0
    left = simpsons_rule(f,a,c)
    right = simpsons_rule(f,c,b)
    if np.max(np.abs(left + right - whole)) <= 15*eps:
        return left + right + (left + right - whole)/15.0
    return recursive_asr(f,a,c,eps/2.0,left) + recursive_asr(f,c,b,eps/2.0,right)


def recursive_asr_imag(f,a,b,eps,whole):
    "Recursive implementation of adaptive Simpson's rule."
    c = (a+b) / 2.0
    left = simpsons_rule(f,a,c)
    right = simpsons_rule(f,c,b)
    if np.max(np.abs(np.imag(left + right - whole))) <= 15*eps:
        return left + right + (left + right - whole)/15.0
    return recursive_asr(f,a,c,eps/2.0,left) + recursive_asr(f,c,b,eps/2.0,right)



def simpson(f,xlim=[0.,1.],eps=0.1):
  """ Integrates a matrix, the measure is the maximun value of the matrix"""
  a = xlim[0]
  b = xlim[1]
  return scalar_asr(f,a,b,eps,simpsons_rule(f,a,b))



def scalar_asr(f,a,b,eps,whole):
    "Recursive implementation of adaptive Simpson's rule."
    c = (a+b) / 2.0
    left = simpsons_rule(f,a,c)
    right = simpsons_rule(f,c,b)
    if np.abs(left + right - whole) <= 15*eps:
        return left + right + (left + right - whole)/15.0
    return scalar_asr(f,a,c,eps/2.0,left) + scalar_asr(f,c,b,eps/2.0,right)



 
def adaptive_simpsons_rule(f,a,b,eps,only_imag=False):
    "Calculate integral of f from a to b with max error of eps."
    if only_imag:
      return recursive_asr_imag(f,a,b,eps,simpsons_rule(f,a,b))
    else:
      return recursive_asr(f,a,b,eps,simpsons_rule(f,a,b))



def random_integrate(f,df=1e-4):
    """Iterate a function until its result shows small enough fluctuations"""
    y0 = f() # compute once
    yt = y0 + 0. # initialize the total result
    yold = y0 + 0. # initialize the old result
    it = 0 # initialize
    while True: # infinite loop
      it += 1 # number of iterations
      y0 = f() # execute the function
      yt += y0 # add contribution
      diff = yold/it - yt/(it+1.) # difference between current and previous
      if np.mean(np.abs(diff))<np.mean(np.max(yt/(it+1.)))*df:
          break
      yold = yt + 0. # update
    return yt/(it+1.) # return result



def peak_integrate(f,x0,x1,xp=0.0,dp=1e-6,**kwargs):
    """Wrapper to quad, for function that has a peak"""
    from scipy.integrate import quad
    i0 = quad(f,x0,xp-dp,**kwargs) # first interval
    i1 = quad(f,xp-dp,xp+dp,**kwargs) # second interval
    i2 = quad(f,xp+dp,x1,**kwargs) # third interval
    return i1+i2+i2
  



def complex_contour(f,xmin=-5,xmax=0,eps=1e-2,mode="upper",
                              **kwargs):
    """Perform an integral using a complex contour"""
#    return integrate_matrix(f,xlim=[xmin,xmax],**kwargs) # integrate matrix
    def fint(x):
        if mode=="upper": 
            z0 = (xmin-xmax)*np.exp(-1j*x*np.pi)/2.
            z = z0 + (xmin+xmax)/2.
            return -1j*(f(z)*z0)*np.pi # integral after change of variables
        elif mode=="lower": 
            z0 = (xmin-xmax)*np.exp(1j*x*np.pi)/2.
            z = z0 + (xmin+xmax)/2.
            return 1j*(f(z)*z0)*np.pi # integral after change of variables
    y = fint(xmin) # evaluate
#    if type(y)=np.array:
    return integrate_matrix(fint,xlim=[0.,1.],eps=eps,**kwargs) # integrate matrix
#    else:
#        import scipy.integrate as integrate
#        return integrate.quad(fint2,0.,1.0,limit=60,epsabs=0.1,epsrel=0.1)[0]
        





