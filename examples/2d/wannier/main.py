import os ;  import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")
from pyqula import wannier
from pyqula import multicell


# now read the Hamiltonian
# you need two files, hamiltonian.wannier and wannier.win
h = wannier.read_multicell_hamiltonian(input_file="hamiltonian.wannier")
#g = geometry.honeycomb_lattice()
#h = g.get_hamiltonian()
hf = multicell.bulk2ribbon(h,n=10)

(k,e,op) = hf.get_bands(operator="xposition")





