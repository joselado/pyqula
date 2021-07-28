# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import chi
import numpy as np
n = 100
g = geometry.chain(n) # chain
g.dimensionality = 0
Bs = np.linspace(0.0,3.0,300)
fo = open("SWEEP.OUT","w")
for B in Bs:
  def ft(r1,r2):
      dr = r1-r2
      dr = dr.dot(dr)
      if 0.9<dr<1.1: return 0.8*np.cos(r1[0]*B*np.pi) + 1.0
      return 0.0
  h = g.get_hamiltonian(fun=ft,has_spin=False)
  
  es = np.linspace(0.0,7.0,200)
  cout = []
  for i in range(10,n-10):
    es,cs = chi.chargechi(h,es=es,i=i,j=i)
    cout.append(cs)
  cs = es*0.0 +0j
  for o in cout: cs += o
  cs /= len(cout)
  for (ie,ic) in zip(es,cs):
      fo.write(str(B)+"   ")
      fo.write(str(ie)+"   ")
      fo.write(str(abs(ic.imag))+"\n")
  fo.flush()
fo.close()







