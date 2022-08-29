# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import specialhamiltonian
h = specialhamiltonian.multilayer_graphene(l=[0,1],ti=0.2)
h = h.get_supercell(3)
h.add_kekule(lambda r: (r[2]>0)*0.1) # one kekule for the upper
h.add_kekule(lambda r: (r[2]<0)*0.2) # anther kekule for the lower
h.get_bands(operator="zposition") # calculate band structure







