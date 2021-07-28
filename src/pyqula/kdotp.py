from __future__ import print_function

def effective_hamiltonian(h,k=[0.,0.,0.],ewindow=None):
  """Return an effective k dot p Hamiltonian at kpoint k"""
  if h.dimensionality != 2: raise # only for 2d




def fermi_velocity(h,n,k=[0.,0.,0.]):
    """Return the Fermi velocity at the Fermi energy"""
    from current import current_operator



