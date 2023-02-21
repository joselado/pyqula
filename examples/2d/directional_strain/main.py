# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
from pyqula import strain
import numpy as np
import matplotlib.pyplot as plt
phis = np.linspace(0.,0.5,6,endpoint=True)
plt.rcParams['figure.figsize'] = [3*len(phis), 3]

iphi = 0 # counter
for phi in phis: # loop over strain directions
    h = geometry.triangular_lattice().get_hamiltonian()
    d = [np.cos(np.pi*phi),np.sin(np.pi*phi),0.] # angle for the strain
    strain.uniaxial_strain(h,d=d,s=0.9) # add strain
    #h.get_multi_fermi_surface(nk=50,energies=np.linspace(-4,4,100),delta=2e-1)
    nk=100 # number of kpoints
    (kx,ky,fs) = h.get_fermi_surface(nk=nk,e=4.0,delta=6e-1)
    fs2d = fs.reshape((nk,nk))
    plt.subplot(1,len(phis),iphi+1) ; iphi += 1 # subplot
    plt.imshow(fs2d,interpolation="bicubic")
    plt.title("Axis = "+str(np.round(d[0:2],2)))
    plt.xticks([]) ;  plt.yticks([])

plt.tight_layout()
plt.show()



