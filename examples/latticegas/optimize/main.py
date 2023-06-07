# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import geometry
from pyqula import supercell
from pyqula import latticegas
import numpy as np
g = geometry.triangular_lattice() # generate the geometry
g = supercell.turn_orthorhombic(g) # make a orthorombic cell
g = g.get_supercell(10)
#g.dimensionality = 0
csa = []
nit = 20
lg = latticegas.LatticeGas(g,filling=1./3.)
lg.add_interaction(Jij=[1.,1.,.0])
lg.optimize_energy(temp=0.1,ntries=1e5)

for i in range(nit):
  lg.set_filling(1./3.)
  lg.optimize_energy(temp=1.,ntries=1e4)
  (rs,csi) = lg.get_correlator()
  csa.append(csi)

csa = np.array(csa)
cs = np.mean(csa,axis=0) # average
dcs = [np.sqrt(np.mean((cs[i]-csa[:,i])**2))/np.sqrt(nit) for i in range(len(cs))] # fluctuation

import matplotlib.pyplot as plt
plt.subplot(1,2,1)
plt.errorbar(rs,cs,yerr=dcs)

z = lg.den
plt.subplot(1,2,2)
plt.scatter(g.x,g.y,c=z,marker="o",cmap="bwr",s=30)
plt.axis("equal")
plt.show()
