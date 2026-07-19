# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")




from pyqula import geometry
from pyqula import topology
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian()
h.add_haldane(0.1) # open a Haldane gap

# Default: sum the Berry curvature over a fixed k-point mesh
c_grid = topology.chern(h,nk=20)
print("Chern number (k-point mesh) is ",c_grid)

# Alternative: integrate the Berry curvature over the BZ with qutecipy
# (tensor cross interpolation + Gauss-Kronrod quadrature) instead of a mesh
c_qtci = topology.chern(h,integration="qtci",nk=20)
print("Chern number (qutecipy integration) is ",c_qtci)
