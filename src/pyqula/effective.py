
import multicell
import scipy.linalg as lg

def effective2d(hin,k0 = 0.0,ewindow = [-1.0,1.0]):
  """Calculates the effective Hamiltonian for a 2d system"""
  h = multicell.turn_multicell(hin) # multicell form
  hkgen = h.get_hk_gen() # get hamiltonian generator
  (es,ws) = lg.eigh(hkgen(k0)) # initial waves
  ws = np.transpose(ws) # transpose the waves
  wf0 = [] # list to store waves
  for i in range(len(es)): # loop over energies
    if ewindow[0]<ewindow[1]: # check whether in window
      wf0.append(ws[i]) # store this wave
  # now calculate the effective Hamiltonian 
  raise
  for order in orders: # loop over orders
    dh = multicell.derivative(h,k0,order=order) # derivative of the hamiltonian

