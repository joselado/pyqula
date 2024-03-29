# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import geometry
from pyqula.heterostructures import LocalProbe
import numpy as np
import matplotlib.pyplot as plt
g = geometry.chain()
#g = geometry.single_square_lattice()
h = g.get_hamiltonian()
#h.add_onsite(2.0)
#h.add_exchange([0.,0.,0.6])
#h.add_rashba(0.6)
h.add_swave(0.1)
#h.get_bands() ; exit()
lp = LocalProbe(h,delta=1e-3) # create a local probe object
#lp.mode = "surface" # mode of the calculation
#lp.bulk_delta = 1e-7 # this is the selfenergy of the bulk
es = np.linspace(-0.2,0.2,101)
lp.T = 0.2 
from pyqula import parallel
parallel.cores = 5
#ts = parallel.pcall(lambda e: lp.didv(energy=e),es)
#ts = parallel.pcall(lambda e: lp.get_kappa(energy=e),es)
#ts = [lp.didv(energy=e) for e in es]
ts = [lp.get_kappa(energy=e) for e in es]
#exit()
import matplotlib.pyplot as plt
plt.plot(es,ts)
plt.ylim([0.,max(ts)])
plt.show()

