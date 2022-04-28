# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


import numpy as np
from pyqula import specialgeometry
g = specialgeometry.multilayer_graphene(l=[0,0]) # bilayer in AA stacking
from pyqula.specialhopping import NNG,ILG # nearest neighbor generator
# get the generator for 1st and 2nd neighbor hoppings
tij_top = NNG(g,[1.0,0.2]).apply(lambda r: r[2]>0.5) 
tij_bottom = NNG(g,[1.0,0.4]).apply(lambda r: r[2]<-0.5) 
til = ILG(g,0.2) # get generator of interlayer hoppings
#tij_top = tij.apply(lambda r: r[2]>0.5) # hoppings for the upper layer
#tij_bottom = tij.apply(lambda r: r[2]<-0.5) # hoppings for the bottom layer
#tij_both = tij_top + tij_bottom + til # add all the hoppings
tij_both = tij_top + tij_bottom + til # add all the hoppings
h = g.get_hamiltonian(tij=tij_both) # create hamiltonian of the system
h.get_bands(operator="zposition")



