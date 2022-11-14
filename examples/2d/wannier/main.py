import os ;  import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")
from pyqula import wannier


# now read the Hamiltonian
# you need two files, hr_truncated.dat and wannier.win
h = wannier.read_multicell_hamiltonian(input_file="hr_truncated.dat")
h.get_bands() # compue band structure

