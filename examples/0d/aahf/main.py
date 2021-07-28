# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import potentials
import numpy as np


g = geometry.chain(400) # chain
g.dimensionality = 0
vs = np.linspace(0.0,4.0,30) # potentials
# loop over v's


def discard(w):
  """Discard edge wavefunctions"""
  w2 = np.abs(w)*np.abs(w) # absolute value
  n = len(w)
  if np.sum(w2[0:n//10])>0.5 or np.sum(w2[9*n//10:n])>0.5: return False
  else: return True


fo = open("LAMBDA_VS_V.OUT","w")
lm = [] # empty array
for v in vs: # loop over strengths of the potential
  h = g.get_hamiltonian(has_spin=False) # get the Hamiltonian
  fun = potentials.aahf1d(v=v,beta=0.0) # function with the AAH potential
  h.add_onsite(fun) # add onsite energies
  (es,ls) = h.get_tails(discard=discard) # return the localization length
  lm.append(np.mean(ls)) # store
  # write in file
  fo.write(str(v)+"    ")
  fo.write(str(np.mean(ls))+"\n")
  fo.flush()
fo.close()


import matplotlib.pyplot as plt

plt.plot(vs,lm,marker="o")
plt.xlabel("AAH Onsite")
plt.ylabel("1/(Localization length)")
plt.show()







