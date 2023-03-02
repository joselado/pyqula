# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import specialgeometry
N = 8 # this is an integer controlling the size of the unit cell (twist angle)
# the bigger the integer, the smaller the twist angle
g = specialgeometry.twisted_bilayer(5) # create the basic twisted structure
def sublattice2name(s):
    if s==1: return "B"
    else: return "N"
g.atoms_names = [sublattice2name(s) for s in g.sublattice] # set species
from pyqula.geometry import write_vasp
write_vasp(g,s=1.45,namefile="twisted_hBN.vasp") # write structure in a file



