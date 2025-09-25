
import numpy as np
from .. import algebra
from .. import integration
from .rg import green_renormalization
from .kchain import green_kchain



def bloch_selfenergy(h,nk=100,energy = 0.0, delta = 1e-2,
                         mode="full", # algorithm for integration
                         gtype="bulk", # bulk or surface
                         error=1e-3):
  """ Calculates the selfenergy of a cell defect,
      input is a hamiltonian class"""
  if mode=="adaptative": mode = "adaptive"
  from ..htk.kchain import detect_longest_hopping
  if detect_longest_hopping(h)==1:
      def gr(h):
        """ Calculates G by renormalization"""
        h = h.get_no_multicell()
        ons,hop = h.intra,h.inter
        gf,sf = green_renormalization(ons,hop,energy=energy,nite=None,
                                error=error,info=False,delta=delta)
        return gf,sf
  elif detect_longest_hopping(h)==2:
      from ..htk.kchain import kchain_NNN # extract up to NNN
      def gr(h):
          (ons,t1,t2) = kchain_NNN(h) # return the three matrices
          from ..greentk.dyson import dysonNNN
          gf,sf = dysonNNN(ons,t1,t2,energy=energy,delta=delta,
                  error=error)
          return gf,sf
  else:
      from ..htk.kchain import kchain_LR # extract all
      def gr(h):
          hops = kchain_LR(h) # return all matrices
          from ..greentk.dyson import dysonLR
          gf,sf = dysonLR(hops,energy=energy,delta=delta,
                  error=error)
          return gf,sf
#  else: # too long range hoppings for RG, use full integration
#      mode = "full_adaptive" 
#      print("Changed to full adaptive mode in selfenergy")
  h = h.copy() # make a copy
  h = h.get_dense() # dense Hamiltonian
  hk_gen = h.get_hk_gen()  # generator of k dependent hamiltonian
  # sanity check for surface mode
  if gtype=="surface": mode = "adaptive" # only the adaptive mode
  #######################################
  d = h.dimensionality # dimensionality of the system
  g = h.intra *0.0j # initialize green function
  e = np.matrix(np.identity(g.shape[0]))*(energy + delta*1j) # complex energy
  if mode=="full":  # full integration in the BZ
    if d==1: # one dimensional
      ks = [[k,0.,0.] for k in np.linspace(0.,1.,nk,endpoint=False)]
    elif d==2: # two dimensional
      ks = []
      kk = np.linspace(0.,1.,nk,endpoint=False)  # interval 0,1
      for ikx in kk:
        for iky in kk:
          ks.append([ikx,iky,0.])
      ks = np.array(ks)  # all the kpoints
    else: raise # raise error
    for k in ks:  # loop in BZ
      g += algebra.inv(e - hk_gen(k))  # add green function  
    g = g/len(ks)  # normalize
  #####################################################
  #####################################################
  elif mode=="renormalization":
    if d==1: # full renormalization
      g,s = gr(h)  # perform renormalization
    elif d==2: # two dimensional, loop over k's
      ks = [[k,0.,0.] for k in np.linspace(0.,1.,nk,endpoint=False)]
#      from ..multicell import rotate90
#      h90 = rotate90(h) # rotated Hamiltonian
      for k in ks:  # loop over k in y direction
 # add contribution to green function
        g += green_kchain(h,k=k,energy=energy,delta=delta,
                error=error,only_bulk=True)
#        g += green_kchain(h90,k=k,energy=energy,delta=delta,error=error)
      g = g/len(ks)
    else: raise
  #####################################################
  #####################################################
  elif mode=="adaptive":
    if d==1: # full renormalization
      g,s = gr(h)  # perform renormalization
      if gtype=="surface": g = s.copy() # take the surface one
      elif gtype=="bulk": pass # do nothing
      else: raise
    elif d==2: # two dimensional, loop over k's
      ks = [[k,0.,0.] for k in np.linspace(0.,1.,nk,endpoint=False)]
      if gtype=="surface": ig = 1 # take the surface one
      elif gtype=="bulk": ig = 0 # take the bulk one
      else: raise
      def fint(k):
        """ Function to integrate """
        return green_kchain(h,k=[k,0.,0.],energy=energy,
                delta=delta,error=error,only_bulk=False)[ig]
      # eps is error, might work....
      g = integration.integrate_matrix(fint,xlim=[0.,1.],eps=error)
        # chain in the y direction
    else: raise
  elif mode=="full_adaptive":
    fint = lambda k: algebra.inv(e - hk_gen(k))  # green's function
    if d==1: # adaptive 1D
        g = integration.integrate_matrix(fint,xlim=[0.,1.],eps=error)
    elif d==2: # adaptive 2D
        g = integration.integrate_matrix_2D(fint,xlim=[0.,1.],ylim=[0.,1.],
              eps=.1)
    else: raise # not implemented
  # now calculate selfenergy
  selfenergy = e - h.intra - algebra.inv(g)
  return g,selfenergy


