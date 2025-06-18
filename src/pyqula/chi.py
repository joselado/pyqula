import numpy as np
import scipy.linalg as lg
from . import parallel
from numba import jit
from . import algebra
from .operators import Operator


use_fortran = False


def chargechi(h,i=0,j=0,es=np.linspace(-3.0,3.0,100),delta=0.01,temp=1e-7):
    """Compute charge response function"""
    if h.dimensionality!=0: raise
    hk = h.get_hk_gen() # get generator
    m = hk(0) # get Hamiltonian
    esh,ws = algebra.eigh(m)
    ws = np.transpose(ws)
    if i<0: raise
    if j<0: raise
    out = 0*es + 0j # initialize
    return es,elementchi(ws,esh,ws,esh,es,i,j,temp,delta,out)

@jit(nopython=True)
def elementchi(ws1,es1,ws2,es2,omegas,ii,jj,T,delta,out):
    """Compute the response function"""
    out  = out*0.0 # initialize
    n = len(ws1) # number of wavefunctions
    for i in range(n): # first loop over states
      oi = es1[i]<0.0 # first occupation
      for j in range(n): # second loop over states
          oj = es2[j]<0.0 # second occupation
          fac = ws1[i][ii]*ws2[j][ii] # add the factor
          fac *= np.conjugate(ws1[i][jj]*ws2[j][jj]) # add the factor
          fac *= oi - oj # occupation factor
          out = out + fac*(1./(es1[i]-es2[j] - omegas + 1j*delta))
    return out






def chargechi_row(h,i=0,es=np.linspace(-3.0,3.0,100),delta=1e-6,temp=1e-7):
    """Compute charge response function"""
    if h.dimensionality!=0: raise
    hk = h.get_hk_gen() # get generator
    m = hk(0) # get Hamiltonian
    esh,ws = algebra.eigh(m)
    ws = np.transpose(ws)
    out = [] # store
    if i<0: raise
    f = lambda j: elementchi(ws,esh,ws,esh,es,i+1,j+1,temp,delta)
    out = parallel.pcall(f,range(m.shape[0])) # parallel call
    return np.array(out)






def chargechi_reciprocal(h,i=None,
        es=np.linspace(-4.,4.,200),delta=1e-3,**kwargs):
    """Return the charge susceptibility in reciprocal space"""
    if h.dimensionality!=0: raise
    if i is None: i = h.geometry.get_central(1)[0]
    # compute correlators
    cs = chargechi_row(h,es=es,delta=delta,**kwargs) 
    # get the distances
    dx = np.abs(h.geometry.r[:,0] - h.geometry.r[i,0])
    # now do the interpolation
    from scipy.interpolate import interp1d
    xs = np.linspace(0.,np.max(dx),len(dx))
    cs = np.array([interp1d(xs,ic)(xs) for ic in cs.T]).T
#    if len(cs)!=len(h.geometry.r[:,0]): raise
    freq = np.fft.fftfreq(len(cs[:,0])) # get the frequencies
    # perform the fourier transform
    fcs = np.array([np.fft.fft(ic) for ic in cs.T])
    fo = open("CHIK.OUT","w")
    print(fcs.shape,es.shape,freq.shape)
    for i in range(len(freq)):
      for j in range(len(es)):
#          fo.write(str(h.geometry.r[i,0])+"  ")
          fo.write(str(freq[i])+"  ")
          fo.write(str(es[j])+"  ")
          fo.write(str(np.abs(fcs[j][i]))+"\n")
    fo.close()



def chiAB(h,q=None,nk=60,**kwargs):
    """Return the generalized response"""
    if q is not None: # q point is provided
        return chiAB_q(h,q=q,nk=nk,**kwargs)
    else:
        qs = h.geometry.get_kmesh(nk=nk) # get the kmesh
        out = [chiAB_q(h,q=q,**kwargs) for q in qs] # get all the k kpoints
        return out[0][0],np.mean([o[1] for o in out],axis=0)



def chiAB_q(h,energies=np.linspace(-3.0,3.0,100),q=[0.,0.,0.],nk=60,
               delta=0.1,temp=1e-7,A=None,B=None,projs=None,
               mode="matrix"):
    """Compute AB response function
       - energies: energies of the dynamical response
       - q: q-vector of the response
       - nk: number of k-points for the integration
       - delta: imaginary part
       - A: first operator
       - B: second operator
       - projs: projection operators
       - mode: response to compute"""
    hk = h.get_hk_gen() # get generator
    if A is None or B is None:
        A = np.identity(h.intra.shape[0],dtype=np.complex128)
        B = A # initial operator
    if type(A)==Operator: A = algebra.todense(A.get_matrix())
    if type(B)==Operator: B = algebra.todense(B.get_matrix())
    # generate the projectors
    if projs is None:
        from . import operators
        projs = [operators.index(h,n=[i]) for i in range(len(h.geometry.r))]
    def getk(k):
        m1 = hk(k) # get Hamiltonian
        es1,ws1 = algebra.eigh(m1)
        ws1 = np.array(ws1.T,dtype=np.complex128)
        m2 = hk(k+q) # get Hamiltonian
        es2,ws2 = algebra.eigh(m2)
        ws2 = np.array(ws2.T,dtype=np.complex128)
        def getAB(Ai,Bj): # compute for a single operator
            out = 0*energies + 0j # initialize
            return chiAB_jit(ws1,es1,ws2,es2,energies,Ai,Bj,temp,delta,out)
        if mode=="matrix": # return a matrix
            out = np.array([[getAB(pi@A,pj@B) for pi in projs] for pj in projs])
            return np.transpose(out,(2,0,1)) # return array of matrices
        elif mode=="trace": # return the trace
            out = np.array([getAB(pi@A,pi@B) for pi in projs])
            return np.mean(out,axis=0) # sum over the first axis
        else: raise # not implemented
    ks = h.geometry.get_kmesh(nk=nk) # get the kmesh
    out = np.mean([getk(k) for k in ks],axis=0) # sum over kpoints
    return energies,out


def chiAB_trace(h,**kwargs):
    """Compute the trace of chiAB"""
    return chiAB(h,mode="trace",**kwargs)




def chiABmap(h,energies=np.linspace(-3.0,3.0,100),nq=30,
                qpath=None,**kwargs):
    """Return the map for the response function"""
    if qpath is None: qpath = h.geometry.get_default_kpath(nk=nq)
    return None



@jit(nopython=True)
def chiAB_jit(ws1,es1,ws2,es2,omegas,A,B,T,delta,out):
    """Compute the response function"""
    out  = out*0.0 # initialize
    n = len(ws1) # number of wavefunctions
    for i in range(n):
      if es1[i]<0.0: oi = 1.0 # first occupation
      else: oi = 0.0
      for j in range(n):
          if es2[j]<0.0: oj = 1.0 # second occupation
          else: oj = 0.0
          fac = np.conjugate(ws1[i]).dot(A@ws2[j]) # add the factor
          fac *= np.conjugate(ws2[j]).dot(B@ws1[i]) # add the factor
          fac *= oi - oj # occupation factor
          out = out + fac*(1./(es1[i]-es2[j] - omegas + 1j*delta))
    return out



from .chitk.static import chargechi as static_charge_correlator
from .chitk.static import szchi as static_sz_correlator
from .chitk.static import sxchi as static_sx_correlator
from .chitk.static import sychi as static_sy_correlator

from .chitk.pmchi import pmchi

