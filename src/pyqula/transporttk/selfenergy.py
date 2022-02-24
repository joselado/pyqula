import numpy as np
from ..green import green_renormalization
from ..algebra import dagger


def get_selfenergy(self,energy,lead=0,delta=None,pristine=False):
   """Return self energy of iesim lead"""
   if delta is None:  delta = self.delta
# if the interpolation function has been created
   if self.interpolated_selfenergy:
       raise # this is not maintained anymore
       if lead==0: return np.matrix(self.selfgen[0](energy)) # return selfenergy
       if lead==1: return np.matrix(self.selfgen[1](energy)) # return selfenergy
# run the calculation
   else:
       if lead==0:
           intra = self.left_intra
           inter = self.left_inter
           if pristine: cou = self.left_inter
           else: cou = self.left_coupling*self.scale_lc
           deltal = delta + self.extra_delta_left # new delta left
       elif lead==1:
           intra = self.right_intra
           inter = self.right_inter
           if pristine: cou = self.right_inter
           else: cou = self.right_coupling*self.scale_rc
           deltal = delta + self.extra_delta_right # new delta right
       else: raise # not implemented
       ggg,gr = green_renormalization(intra,inter,energy=energy,delta=deltal)
       selfr = cou@gr@dagger(cou) # selfenergy
       return selfr # return selfenergy



