# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
import numpy as np
from pyqula import geometry
from pyqula import scftypes
from pyqula import operators
from scipy.sparse import csc_matrix


g = geometry.honeycomb_lattice() # create a honeycomb lattice
h0 = g.get_hamiltonian() # create hamiltonian of the system



ps = np.linspace(0.,.1,10) # create the different KM couplings
#ps = [0.1]
es0 = [] # empty list for the total energies
es1 = [] # empty list for the total energies


f = open("ENERGY_VS_ANGLE.OUT","w") # file with the results
for p in ps: # loop over angles
  h = h0.copy() # copy Hamiltonian
  h.add_kane_mele(p) # Add Kane-Mele SOC
  hoff = h.copy()
  hin = h.copy()
  # Hamiltonian with in-plane quantization axis
  hin.global_spin_rotation(angle=0.5,vector=[1.,0.,0.])
  U = 3.0 # large U to get antiferromagnetism
  # offplane antiferro initialization
  mfoff = scftypes.guess(h,mode="antiferro",fun = lambda x: [0.,0.,1.0]) 
  # inplane antiferro initialization
  mfin = scftypes.guess(h,mode="antiferro",fun = lambda x: [1.0,0.,0.]) 
  # compute the energies using collinear formalism with different
  # quantization axis. In this case, the collinear formalism
  # will enforce the magnetization in the z direction
  scfoff = scftypes.hubbardscf(hoff,nkp=5,filling=0.5,g=U,
                mix=0.5,mf=mfoff,collinear=True)
  scfin = scftypes.hubbardscf(hin,nkp=5,filling=0.5,g=U,
                mix=0.5,mf=mfoff,collinear=True)
#  print(scfoff.total_energy)
#  print(scfin.total_energy)
#  exit()
  # alternatively, we can use a non-collinear formalism and go to the
  # ground state with using two different initializations
  # compute energies using non-collinear formalism but different initialization
  scfoff2 = scftypes.hubbardscf(h,nkp=5,filling=0.5,g=U,
                mix=0.5,mf=mfoff,collinear=False)
  scfin2 = scftypes.hubbardscf(h,nkp=5,filling=0.5,g=U,
                mix=0.5,mf=mfin,collinear=False)
  # Now compute energy differences
  e0 = scfoff.total_energy - scfin.total_energy
  e1 = scfoff2.total_energy - scfin2.total_energy
  # and store the energies
  es0.append(e0) # store energy
  es1.append(e1) # store energy
  f.write(str(p)+"   "+str(e0)+"    "+str(e1)+"\n") # write in the file
f.close() # close the file


## Plot the energies ##

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['font.family'] = "Bitstream Vera Serif"
fig = plt.figure()
fig.subplots_adjust(0.2,0.2)
plt.plot(ps,es1,c="red",marker="o",label="non-collinear")
plt.plot(ps,es0,c="blue",label="collinear")
plt.legend()
plt.xlabel("Kane-Mele coupling")
plt.ylabel("Anisotropy energy")
plt.show()








