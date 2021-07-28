# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
import numpy as np
from pyqula importgeometry
from pyqula importscftypes
import operators
from scipy.sparse import csc_matrix
g = geometry.honeycomb_zigzag_ribbon(4) # create geometry of a zigzag ribbon
g = geometry.square_lattice()
#g = geometry.honeycomb_lattice_C6()
g = geometry.honeycomb_lattice()
g = geometry.chain()
g = g.supercell(10)
g.write()
#exit()
h = g.get_hamiltonian() # create hamiltonian of the system
h = h.get_multicell()
#h.shift_fermi(0.5)
def delta(r1,r2):
  dr = r1-r2
  if 0.1<dr[0]<1.1: return [-0.5,-0.5,0.]
  elif -1.1<dr[0]<-0.1: return [0.5,0.5,0.]
  else: return [0.,0.,0.]
#h.add_pwave(delta)
#print(h.hopping[1].m)
#h.get_bands()
#exit()
#h.add_rashba(0.5)
#h.add_zeeman(0.5)
#h.shift_fermi(-1.0)
h.add_swave(0.0)
mf = scftypes.guess(h,mode="swave",fun=lambda r: 0.02)
#h.remove_spin()
mode = {"U":-2}
mode = {"V":-1.}
#h.turn_sparse()
scf = scftypes.selfconsistency(h,nkp=100,g=-1.0,
              mix=0.9,mf=mf,mode=mode,energy_cutoff=None)
#scf = scftypes.selfconsistency(h,nkp=40,filling=0.5,g=-6.0,
#              mix=0.9,mf=scf.mf,mode=mode)
h = scf.hamiltonian # get the Hamiltonian
#print(csc_matrix(np.round(h.intra,3)))
#op = operators.get_sz(h)
#print(op.shape)
#h.add_zeeman([0.,0.,4])
h.get_bands() # calculate band structure
#print(scf.magnetization)
#h.write()
import groundstate
#groundstate.swave(h)
#groundstate.hopping(h)







