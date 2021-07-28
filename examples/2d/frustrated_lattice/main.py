# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula importgeometry
import topology
import klist
import classicalspin
import films
import ribbon
import islands
g = geometry.honeycomb_lattice() # generate the geometry
g = geometry.triangular_lattice() # generate the geometry
g = geometry.pyrochlore_lattice() # generate the geometry
#g = geometry.kagome_lattice() # geenrate the geometry
#g = geometry.diamond_lattice_minimal() # generate the geometry
g = films.geometry_film(g,nz=2)
#g = g.remove(len(g.r)-1)
#g = ribbon.bulk2ribbon(g,n=10)
g = g.supercell(2)
#g = islands.get_geometry(name="triangular",n=3)
#g.dimensionality = 0
#g.write()
#g.write()
#g = geometry.kagome_lattice() # geenrate the geometry
#g = g.supercell(2)
sm = classicalspin.SpinModel(g) # generate a spin model
sm.add_heisenberg() # add heisenber exchange
#sm.add_field([0.,0.,1.5])
sm.minimize_energy() # minimize Hamiltonian
#exit()
h = g.get_hamiltonian() # get the Hamiltonian
h.add_magnetism(sm.magnetization*2.0) # add magnetization
h.write_magnetization()
def ffer(r): return r[2]*7
#h.shift_fermi(ffer) # electric field
#h.add_kane_mele(1.0)
#h.add_antiferromagnetism(5.0)
h = ribbon.hamiltonian_ribbon(h,n=10)
h.turn_dense()
h.get_bands(operator="yposition")
#h.get_bands()
sm.write()







