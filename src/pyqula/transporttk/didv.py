from ..parallel import pcall
import numpy as np
from ..integration import simpson
from scipy.integrate import quad
from .. import parallel



def generic_didv(self,temp=0.,**kwargs):
    """Wrapper to compute the dIdV at finite temperature"""
    if temp==0.: return zero_T_didv(self,**kwargs) # zero temperature
    else: # finite temperature
        return finite_T_didv(self,temp=temp,**kwargs)


def finite_T_didv(self,temp,energy=0.0,**kwargs):
    """Finite temperature dIdV"""
#    return zero_T_didv(self,energy=energy,**kwargs)
    from .fermidirac import fermidirac as FD
    dt = 20 # max T range
    de = temp # energy difference to compute the derivative
    ### Use simpson integration
    def f(e):
        out = zero_T_didv(self,energy=energy+e,**kwargs) 
        out *= FD(e-de,temp=temp) - FD(e+de,temp=temp) 
        return out/de
    from ..integration import peak_integrate
#    return peak_integrate(f,-dt*temp,dt*temp,xp=0.0,dp=temp,
#                           epsrel=1e-2,limit=10)[0]/2.
    return quad(f,-dt*temp,dt*temp,epsrel=1e-4,limit=60)[0]/2.


def zero_T_didv(self,delta=None,**kwargs):
    """Zero temperature dIdV"""
    if delta is None: delta = self.delta # set the own delta
    if self.dimensionality==1: # one dimensional
        return zero_T_didv_1D(self,**kwargs)
    elif self.dimensionality==2: # two dimensional
        return zero_T_didv_2D(self,**kwargs)
    else: raise





def zero_T_didv_1D(self,energy=0.0,delta=None,**kwargs):
    """Wrapper for the dIdV in one dimension"""
    if delta is None: delta = self.delta # set the own delta
    if not self.dimensionality==1: raise # only for one dimensional
    return didv(self,energy=energy,delta=delta,**kwargs)
    try:
        return didv(self,energy=energy,delta=delta,**kwargs)
    except:
        print("Something wrong in didv, returning 0")
        return 1e-10


def zero_T_didv_2D(self,energy=0.0,delta=None,nk=10,
                   imode="quad",**kwargs):
    """Wrapper for the dIdV in two-dimensions"""
    if delta is None: delta = self.delta # set the own delta
    if not self.dimensionality==2: raise # only for two dimensional
    # function to integrate
    print("Computing",energy)
    f = lambda k: self.generate(k,self.scale_lc,self.scale_rc).didv(energy=energy,delta=delta,**kwargs)
    if imode=="grid":
        out = pcall(f,np.linspace(0.,1.,nk,endpoint=False))
        return np.trapz(out,dx=1./nk)
    elif imode=="simpson":
        return simpson(f,eps=1e-4,xlim=[0.,1.])
    elif imode=="quad":
        return quad(f,0.,1.,epsrel=1e-1,limit=20)[0]
    else: 
        print("Unrecognized imode")
        raise



from .smatrix import get_smatrix
from .. import algebra
dagger = algebra.dagger



def didv(ht,energy=0.0,delta=1e-6,kwant=False,opl=None,opr=None,**kwargs):
    """Calculate differential conductance"""
    if ht.has_eh: # for systems with electons and holes
        s = get_smatrix(ht,energy=energy,check=True) # get the smatrix
        r1,r2 = s[0][0],s[1][1] # get the reflection matrices
        get_eh = ht.get_eh_sector # function to read either electron or hole
        # select the normal lead
        # r1 is normal
        r = ht.get_reflection_normal_lead(s) # return the reflection
        ree = get_eh(r,i=0,j=0) # reflection e-e
        reh = get_eh(r,i=0,j=1) # reflection e-h
        Ree = np.trace(dagger(ree)@ree) # total e-e reflection 
        Reh = np.trace(dagger(reh)@reh) # total e-h reflection 
        G = (ree.shape[0] - Ree + Reh).real # conductance
        return G
    else:
        if kwant:
          if opl is not None or opr is not None: raise # not implemented
          from . import kwantlink
          return kwantlink.transport(ht,energy)
        s = get_smatrix(ht,energy=energy) # get the smatrix
        if opl is not None or opr is not None: # some projector given
          raise # this does not make sense
          U = [[np.identity(s[0][0].shape[0]),None],
                   [None,np.identity(s[1][1].shape[0])]] # projector
          if opl is not None: U[0][0] = opl # assign this matrix
          if opr is not None: U[1][1] = opr # assign this matrix
          #### those formulas are not ok
          s[0][0] = U[0][0]@s[0][0]@U[0][0] # new smatrix
          s[0][1] = U[0][0]@s[0][1]@U[1][1] # new smatrix
          s[1][0] = U[1][1]@s[1][0]@U[0][0] # new smatrix
          s[1][1] = U[1][1]@s[1][1]@U[1][1] # new smatrix
        r1,r2,t = s[0][0],s[1][1],s[0][1] # get the reflection matrices
        # select a normal lead (both of them are)
        # r1 is normal
        ree = r1
        Ree = np.trace(dagger(ree)@ree) # total e-e reflection 
        G1 = (ree.shape[0] - Ree).real # conductance
        G2 = np.trace(s[0][1]@dagger(s[0][1])).real # total e-e transmission
        return (G1+G2)/2.







