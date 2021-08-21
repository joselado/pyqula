from ..heterostructures import create_leads_and_central_list
from ..heterostructures import Heterostructure


def build(h1,h2,central=None,**kwargs):
  """Create a heterostructure, works also for 2d"""
#  if central is None: central = [h1,h2] # list
  if central is None: central = [] # list
  if h1.dimensionality==1: # one dimensional
    return create_leads_and_central_list(h2,h1,central,**kwargs) # standard way
  elif h1.dimensionality==2:  # two dimensional
    def fun(k,lc,rc):
      # evaluate at a particular k point
      h1p = h1.get_1dh(k)
      h2p = h2.get_1dh(k)
      centralp = [hc.get_1dh(k) for hc in central]
      out = create_leads_and_central_list(h1p,h2p,centralp,**kwargs) # standard way
      out.scale_lc = lc
      out.scale_rc = rc
      return out # return 1d heterostructure
    hout = Heterostructure() # create
    hout.dimensionality = 2 # two dimensional
    hout.generate = fun # function that generates the heterostructure
    return hout # function that return a heterostructure
  else: raise

