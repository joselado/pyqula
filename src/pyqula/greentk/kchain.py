from .rg import green_renormalization
from .. import algebra

def green_kchain_NN(h,k=0.,energy=0.,delta=0.01,only_bulk=True,
                    error=0.0001,hs=None,**kwargs):
    """ Calculates the green function of a kdependent chain for a 2d system """
    def gr(ons,hop):
      """ Calculates GF by renormalization"""
      gf,sf = green_renormalization(ons,hop,energy=energy,nite=None,
                              error=error,info=False,delta=delta)
      if hs is not None: # surface matrix provided
        ez = (energy+1j*delta)*np.identity(h.intra.shape[0]) # energy
        sigma = hop@sf@algebra.dagger(hop) # selfenergy
        if callable(hs): ons2 = hs(k)
        else: ons2 = hs
        sf = algebra.inv(ez - ons2 - sigma) # return Dyson
      if only_bulk:  return gf
      else:  return gf,sf
    (ons,hop) = get1dhamiltonian(h,k,**kwargs) # get 1D Hamiltonian
    return gr(ons,hop)  # return green function


def green_kchain(h,**kwargs):
    from ..htk.kchain import detect_longest_hopping
#    return green_kchain_LR(h,**kwargs)  # return green function
    if detect_longest_hopping(h)==1: # only NN
        return green_kchain_NN(h,**kwargs)  # return green function
    elif detect_longest_hopping(h)==2: # up to NNN
        return green_kchain_NNN(h,**kwargs)  # return green function
    else: # generic case
        return green_kchain_LR(h,**kwargs)  # return green function



def green_kchain_NNN(H,k=[0.,0.,0.],**kwargs):
    """Return Green function when there is long range hopping"""
    from ..htk.kchain import detect_longest_hopping
    if detect_longest_hopping(H)>2: raise # up to NNN
    from ..htk.kchain import kchain_NNN # extract up to NNN
    (ons,t1,t2) = kchain_NNN(H,k=k) # return the three matrices
    from ..greentk.dyson import dysonNNN
    return dysonNNN(ons,t1,t2,**kwargs)


def green_kchain_LR(H,k=[0.,0.,0.],**kwargs):
    """Return Green function when there is long range hopping"""
    from ..htk.kchain import kchain_LR # extract up to NNN
    hops = kchain_LR(H,k=k) # return the three matrices
    from ..greentk.dyson import dysonLR
    return dysonLR(hops,**kwargs)






def get1dhamiltonian(hin,k=[0.0,0.,0.],reverse=False,**kwargs):
  """Return onsite and hopping matrix for a 1D Hamiltonian"""
  from .. import multicell
  (ons,hop) = multicell.kchain(hin,k=k,**kwargs)
  if reverse: return (ons,algebra.dagger(hop)) # return 
  else: return (ons,hop) # return 




#def green_kchain_evaluator(h,k=0.,delta=0.01,only_bulk=True,
#                    error=0.0001,hs=None,reverse=False):
#  """ Calculates the green function of a kdependent chain for a 2d system """
#  def gr(ons,hop,energy):
#    """ Calculates G by renormalization"""
#    gf,sf = green_renormalization(ons,hop,energy=energy,nite=None,
#                            error=error,info=False,delta=delta)
#    if hs is not None: # surface matrix provided
#      ez = (energy+1j*delta)*np.identity(h.intra.shape[0]) # energy
#      sigma = hop@sf@algebra.dagger(hop) # selfenergy
#      if callable(hs): ons2 = ons + hs(k)
#      else: ons2 = ons + hs
#      sf = algebra.inv(ez - ons2 - sigma) # return Dyson
#    # which green function to return
#    if only_bulk:  return gf
#    else:  return gf,sf
#  (ons,hop) = get1dhamiltonian(h,k,reverse=reverse) # get 1D Hamiltonian
#  def fun(energy): # evaluator
#    return gr(ons,hop,energy)  # return green function
#  return fun # return the function


def green_kchain_evaluator(h,**kwargs):
    def fun(energy): # evaluator
        return green_kchain(h,energy=energy,**kwargs)
    return fun
