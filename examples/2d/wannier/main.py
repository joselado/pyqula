import os ;  import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")
from pyqula import wannier


# now read the Hamiltonian
# you need two files, hamiltonian.wannier and wannier.win
h = wannier.read_multicell_hamiltonian(input_file="hamiltonian.wannier")
h.get_bands() # compute band structure





