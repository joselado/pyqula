# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
import numpy as np
from pyqula import geometry
from pyqula import meanfield

g = geometry.square_lattice() # geometry for a square lattice
Us = np.linspace(0.,30.,40) # different Us


def fimpurity(ri):
    """This function is 1 for the impurity, 0 elsewhere"""
    dr = ri - g.r[0]
    if dr.dot(dr)<1e-3: return 1.0
    else: return 0.0


def get(U):
    """Perform a selfconsistent calculation"""
    print(U)
    h = g.get_hamiltonian() # create hamiltonian of the system
    mf = meanfield.guess(h,mode="ferro") # ferro initialization
    # perform SCF with specialized routine for Hubbard
    fU = lambda r: U*fimpurity(r) # Hubbard for the site
    scf = meanfield.Vinteraction(h,nk=20,
                  filling=0.5, # set at half filling
                  U=fU, # spatially dependent U
                  mix = 0.9, # quite agressive mixing
                  mf=mf, # initial guess
                  constrains=["no_charge"] # ignore chemical potential renorm
                  )
    h = scf.hamiltonian # selfconsistent Hamiltonian
    return h.compute_vev("sz")[0] # magnetization of the Hubbard site

ms = [get(U) for U in Us]
np.savetxt("M_VS_U.OUT",np.array([Us,np.abs(ms)]).T)







