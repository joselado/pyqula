# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Quadrupole moment of the Benalcazar-Bernevig-Hughes model from the nested
# Wilson loops, across the transition at delta=0 where the bulk gap closes

import numpy as np
from pyqula import specialhamiltonian
from pyqula import topology

ds = np.linspace(-0.8,0.8,9) # hopping imbalance, 1-delta inside the cell
qs,ps,gaps = [],[],[]
for d in ds:
    if abs(d)<1e-6: # the bulk gap closes at delta=0
        qs.append(np.nan) ; ps.append(np.nan) ; gaps.append(0.) ; continue
    h = specialhamiltonian.square_2OTI(delta=d) # quadrupole model
    qs.append(h.get_quadrupole_moment(nk=30)) # quadrupole moment
    ps.append(h.get_wannier_sector_polarization(nk=30)) # sector polarization
    gaps.append(topology.wannier_gap(h,nk=30)) # distance of the Wannier bands to 0, 1/2
    print("delta",round(d,2),"q_xy",round(qs[-1],6),"p",round(ps[-1],6),
            "Wannier gap",round(gaps[-1],4))

import matplotlib.pyplot as plt
plt.plot(ds,qs,marker="o",label="$q_{xy}$")
plt.plot(ds,gaps,marker="s",label="Wannier gap")
plt.xlabel("$\\delta$") ; plt.legend()
plt.show()
