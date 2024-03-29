# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import specialgeometry
N = 5 # this is an integer controlling the size of the unit cell (twist angle)
# the bigger the integer, the smaller the twist angle
g = specialgeometry.twisted_bilayer(N) # create the basic twisted structure

# if you want an orthorombic unit cell, uncomment these lines
from pyqula.supercell import turn_orthorhombic
g = turn_orthorhombic(g) # build an orthorhombic one

def sublattice2name(s):
    if s==1: return "B"
    else: return "N"
g.atoms_names = [sublattice2name(s) for s in g.sublattice] # set species
from pyqula.geometry import write_vasp
write_vasp(g,s=1.45,namefile="twisted_hBN.vasp") # write structure in a file



