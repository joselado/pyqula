# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula importgeometry
import numpy as np
n = 40
g = geometry.chain(n) # chain
g.dimensionality = 0
Bs = np.linspace(0.0,2.0,30)
fo = open("SWEEP.OUT","w")
#Bs = [0.3]
for B in Bs:
#  B = 1.0
  cout = []
  for phi in np.linspace(0.,2,30):
    def ft(r1,r2):
        dr = r1-r2
        dr = dr.dot(dr)
        rm = (r1+r2)/2.
        if 0.9<dr<1.1: return 0.8*np.cos((rm[0]*B+phi)*np.pi) + 1.0
        return 0.0
    h = g.get_hamiltonian(fun=ft,has_spin=False)
    
    es = np.linspace(0.0,10.0,600)
from pyqula importscftypes
  #  V = 0.5
  #  mf = scftypes.selfconsistency(h,g=V,mode="V",mf=h.intra*0.0)
  #  h = mf.hamiltonian
from pyqula importdos
  #  cs = dos.dos0d(h,es=es,i=[0])
    import chi
  #  cout = []
  #  es,cs = chi.chargechi(h,es=es,i=20,j=20) ; cs = cs.imag
    es,cs = chi.chargechi(h,es=es,i=0,j=0)
    cout.append(cs)
  cs = es*0.0 +0j
  for o in cout: cs += o
  cs /= len(cout)
  for (ie,ic) in zip(es,cs):
      fo.write(str(B)+"   ")
      fo.write(str(ie)+"   ")
      fo.write(str(abs(ic.real))+"\n")
  fo.flush()
fo.close()







