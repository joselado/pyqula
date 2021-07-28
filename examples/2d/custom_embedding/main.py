# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





import numpy as np

def get_bulk_green_function(h0,energy=0.0,eta=1e-3):
  """Function to compute the bulk Green's function"""
  from pyqula import geometry
  from pyqula import green
  g = geometry.honeycomb_lattice()
  h = g.get_hamiltonian()
  h.intra = np.matrix(h0.intra)
  h.tx = np.matrix(h0.tx)
  h.ty = np.matrix(h0.ty)
  h.txy = np.matrix(h0.txy)
  h.txmy = np.matrix(h0.txmy)
  h.is_multicell
  #gf,selfe = green.bloch_selfenergy(h,energy=energy,delta=eta,mode="full",nk=10)
#  gf,selfe = green.bloch_selfenergy(h,energy=energy,delta=eta,mode="renormalization")
  gf,selfe = green.bloch_selfenergy(h,energy=energy,delta=eta,mode="adaptive")
  return gf # return Green's function




if __name__=="__main__":
    class H(): pass # dummy class
    h = H() # dummy object
    # create the Hamiltonian for the honeycomb lattice
    # in the following atributes you can put the matrices
    # of your specific model
    t = np.array([[0.,1.],[0.,0.]])
    h.intra = t + t.T # intracell hopping
    h.tx = t # hopping in a1 direction
    h.ty = t # hopping in a2 direction
    h.txy = 0.*t # hopping in a1+a2 direction
    h.txmy = 0.*t # hopping in a1-a2 direction
    # compute the bulk green's function using the special function
    eta = 1e-2
    gf = get_bulk_green_function(h,energy=0.3,eta=eta)
    import scipy.linalg as lg
    def f(e):
      gf = get_bulk_green_function(h,energy=e,eta=eta)
      selfe = t@gf@t.T # selfenergy
      ons = np.array([[10000.0,0.],[0.,0.]])
#      ons = np.array([[0.0,1.],[1.,0.]])
      g = lg.inv(ons - np.array([[1.,0.],[0.,1.]])*(e-1j*1e-2)- selfe)
      print(e)
      return -np.trace(g).imag
    es = np.linspace(-2.,2.,40)
    ds = [f(e) for e in es]
    np.savetxt("DOS.OUT",np.array([es,ds]).T)










