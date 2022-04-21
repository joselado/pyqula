# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import geometry

import numpy as np

m = np.genfromtxt("bands_TaS2_SOC.txt").T
k,e = m[0],m[1]
k = k - np.min(k)
k = k/(np.max(k)*1) # path is G-K-M

g = geometry.triangular_lattice()
def fit(x):
# create triangular lattice
  from pyqula.specialhamiltonian import TMDC_MX2
  soc = x[1]
  ts = x[2:]
  h = TMDC_MX2(tij=ts,normalize=False,soc=soc)
  h.remove_spin()
  h.turn_dense()
  h.add_onsite(x[0])
  hk = h.get_hk_gen()
  es = np.array([np.trace(hk([ik,ik,0])).real for ik in k])
  print(x)
  return np.sum((e-es)**2)

from scipy.optimize import minimize

x0 = np.random.random(7)
res = minimize(fit,x0)
soc = res.x[1]
ts = res.x[2:]
print(ts,soc)
h = TMDC_MX2(tij=ts,normalize=False,soc=soc)
h.get_bands()
#import matplotlib.pyplot as plt
#plt.scatter(k,e)
#plt.show()







