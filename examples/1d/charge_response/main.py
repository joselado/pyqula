# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
import numpy as np
g = geometry.chain()
g = geometry.honeycomb_armchair_ribbon(1)
h = g.get_hamiltonian()
from pyqula import chi

energies = np.linspace(0.0,4.0,60)
fo = open("CHI.OUT","w")
for q in np.linspace(0.,1.,60):
    print("Doing",q)
    out = chi.chiAB(h,energies=energies,q=[q,0.,0.])
    for (e,o) in zip(energies,out):
      o = np.trace(o)
      fo.write(str(q)+"  ")
      fo.write(str(e)+"  ")
      fo.write(str(o.imag)+"  ")
      fo.write(str(o.real)+"  ")
      fo.write("\n")
fo.close()

