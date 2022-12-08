# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")



from pyqula import geometry
import numpy as np
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian()
def strain_direction(dr):
    """Function to add strain in a direction, in particular the x-direction"""
    dr2 = dr.dot(dr)
    if dr2>1e-3:
      dr = dr/np.sqrt(dr2) # normalize vector
      return 1. + abs(dr[0]) # square component in the x direction
    else: return 1.0 # if it is onsite
h.add_strain(strain_direction,mode="directional")
h.write_hopping()
h.get_fermi_surface(e=0.5)







