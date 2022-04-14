# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")



from pyqula import specialhamiltonian
h = specialhamiltonian.FeSe()
h.geometry.write()
h.get_bands()
h.get_fermi_surface()





