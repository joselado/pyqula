import numpy as np
from ..green import green_renormalization
from ..algebra import dagger


def get_selfenergy(self,energy,lead=0,delta=None,pristine=False):
   """Return self energy of iesim lead"""
   # in case you use a dummy selfenergy
   if self.use_minimal_selfenergy:
       return minimal_selfenergy(self,lead=lead)
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





def minimal_selfenergy(self,lead=0,**kwargs):
    """Function returning a minimal selfenergy"""
    print("Using a dummy metallic selfenergy")
    if not self.use_minimal_selfenergy: raise # make sure this is ok
    gamma = self.minimal_selfenergy_gamma # get the gamma
    # reescale for compatibility with everything else
    if lead==0: 
        scale = self.scale_lc**2
        n = self.left_inter.shape[0]
    elif lead==1: 
        scale = self.scale_rc**2
        n = self.right_inter.shape[0]
    out = np.identity(n,dtype=np.complex) # output matrix
    out = -scale*gamma*out*1j # multiply
    return out

