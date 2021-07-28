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
h0.add_kane_mele(0.05) # Add Kane-Mele SOC



ps = np.linspace(0.,1.,30) # create different angles, from 0 to pi
es = [] # empty list for the total energies


f = open("ENERGY_VS_ANGLE.OUT","w") # file with the results
for p in ps: # loop over angles
  h = h0.copy() # copy Hamiltonian
  # the following will rotate the spin quantization axis in the
  # unit cell, so that we can fix the magnetization along one
  # direction and compute the magnetic anisotropy
  # the angle is given in units of pi, i.e. p=0.5 is 90 degrees
  h.global_spin_rotation(angle=p,vector=[1.,0.,0.])
  U = 3.0 # large U to get antiferromagnetism
  # antiferro initialization
  mf = scftypes.guess(h,mode="antiferro",fun = lambda x: 1.0) 
  # perform SCF with specialized routine for collinear Hubbard
  scf = scftypes.hubbardscf(h,nkp=5,filling=0.5,g=U,
                mix=0.5,mf=mf,collinear=True)
  e = scf.total_energy # get the energy of the system
  es.append(e) # store energy
  f.write(str(p)+"   "+str(e)+"\n") # write in the file
f.close() # close the file


## Plot the energies ##

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['font.family'] = "Bitstream Vera Serif"
fig = plt.figure()
fig.subplots_adjust(0.2,0.2)
plt.plot(ps*180,es,marker="o",c="blue")
plt.xlabel("Angle of the magnetization [degrees]")
plt.ylabel("Total energy")
plt.show()








