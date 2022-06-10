from ..heterostructures import create_leads_and_central_list
from ..heterostructures import Heterostructure
from ..algebra import dagger
import numpy as np


def build(h1,h2,central=None,**kwargs):
  """Create a heterostructure, works also for 2d"""
#  if central is None: central = [h1,h2] # list
  if central is None: central = [] # list
  # make the Hamiltonians compatible
  from ..htk.mode import make_compatible
  h1 = make_compatible(h1,h2)
  h2 = make_compatible(h2,h1)
  central = [make_compatible(hi,h1) for hi in central]   
  ###############
  if h1.dimensionality==h2.dimensionality: # same dimensionality
      if h1.dimensionality==1: # one dimensional
        return create_leads_and_central_list(h2,h1,central,**kwargs) # standard way
      elif h1.dimensionality==2:  # two dimensional
        def fun(k,lc=1.0,rc=1.0):
          # evaluate at a particular k point
          h1p = h1.get_1dh(k)
          h2p = h2.get_1dh(k)
          centralp = [hc.get_1dh(k) for hc in central]
          out = create_leads_and_central_list(h2p,h1p,centralp,**kwargs) # standard way
          out.scale_lc = lc
          out.scale_rc = rc
          return out # return 1d heterostructure
        hout = Heterostructure() # create
        hout.Hr = h2.copy() # store Hamiltonian
        hout.Hl = h1.copy() # store Hamiltonian
        hout.dimensionality = 2 # two dimensional
        hout.generate = fun # function that generates the heterostructure
        return hout # function that return a heterostructure
      else: raise # not implemented
  else:
# this is a temporal fix
      if h1.dimensionality==1 and h2.dimensionality==2: # 1D to 2D
          if len(central)!=0: raise # not implemented
          h2t = h2.get_1dh(0.) # create a 1D Hamiltonian
          HT = build(h1,h2t,central=central,**kwargs) # create a fake HT
          HT0 = HT.copy() # copy the HT
          def get_selfenergy(energy,nk=40,lead=0,delta=HT.delta,**kwargs):
              if lead==0: return HT0.get_selfenergy(energy,lead=lead,
                                          delta=delta,
                                       **kwargs) # default
              elif lead==1:
                  from .. import green
                  gf = green.bloch_selfenergy(h2,energy=energy,
                                         nk=nk,mode="adaptive",
                                         delta=delta)[0]
                  gamma = dagger(HT0.left_inter)
                  return gamma@gf@dagger(gamma)
              else: raise
          HT.get_selfenergy = get_selfenergy # overwrite
          HT.block_diagonal = False # block diagonal
          HT.central_intra = h1.intra # intra cell
          HT.right_coupling = dagger(HT.left_coupling)
          return HT
      raise # not finished



class Hybrid_Heterostructure(Heterostructure):
    """New class for hybrid heterostructures"""
    def __init__(self,h1,h2):
        if h1.dimensionality==1 and h2.dimensionality==2: # 1D to 2D
          if len(central)!=0: raise # not implemented
          h2t = h2.get_1dh(0.) # create a 1D Hamiltonian
          HT = build(h1,h2t,central=central,**kwargs) # create a fake HT
          HT0 = HT.copy() # copy the HT
          def get_selfenergy(energy,nk=40,lead=0,delta=HT.delta,**kwargs):
              if lead==0: return HT0.get_selfenergy(energy,lead=lead,
                                          delta=delta,
                                       **kwargs) # default
              elif lead==1:
                  from .. import green
                  gf = green.bloch_selfenergy(h2,energy=energy,
                                         nk=nk,mode="adaptive",
                                         delta=delta)[0]
                  gamma = dagger(HT0.left_inter)
                  return gamma@gf@dagger(gamma)
              else: raise
          HT.get_selfenergy = get_selfenergy # overwrite
          HT.block_diagonal = False # block diagonal
          HT.central_intra = h1.intra # intra cell
          HT.right_coupling = dagger(HT.left_coupling)
          raise # not finished



def get_reflection_normal_lead(ht,s):
    """Get reflection matrix of the normal lead"""
    r1,r2 = s[0][0],s[1][1] # get the reflection matrices
    get_eh = ht.get_eh_sector # function to read either electron or hole
    # select the normal lead
    # r1 is normal
    if np.sum(np.abs(get_eh(ht.left_intra,i=0,j=1)))<0.0001: r = r1
    elif np.sum(np.abs(get_eh(ht.right_intra,i=0,j=1)))<0.0001: r = r2
    else:
        print("There is SC in both leads, aborting")
        raise
    return r
















