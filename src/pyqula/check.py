from __future__ import print_function
import numpy as np
from scipy.sparse import csc_matrix

def equal(m1,m2,tol=1e-4):
  """Check if two matrices are the same"""
  if np.max(np.abs(m1-m2))>tol:
#    print(csc_matrix(m1-m2))
    print("Maximum difference",np.max(np.abs(m1-m2)))
#    print("\n")
#    print(csc_matrix(m2))
    return False
  else: return True





def check_hermitian(h,tol=1e-5):
  hk = h.get_hk_gen() # get generator
  m = hk(np.random.random(3)) # random k-point
  if not equal(m,np.conjugate(m).T,tol=tol):
    raise ValueError("the Hamiltonian is not Hermitian, the largest "
            "deviation of h(k) from its adjoint is "
            +str(np.max(np.abs(m-np.conjugate(m).T))))





def check_hamiltonian(h,tol=1e-5):
  """Do various checks in Hamiltonian, to ensure that nothing weird happens"""
  hk = h.get_hk_gen() # get generator
  if not h.non_hermitian: check_hermitian(h,tol=tol)
  if h.has_eh: # if it has electron hole degree of freedom
    v = np.random.random(3) # random kpoint
    m1 = hk(v) # Hamiltonian
    m2 = hk(-v) # Hamiltonian in time reversal point
    from .superconductivity import eh_operator
    eh = eh_operator(m1) # get the function
    if not equal(m1,-eh(m2),tol=tol): 
      raise ValueError("the Hamiltonian does not have electron-hole "
              "symmetry, the largest deviation of h(k) from -eh(h(-k)) is "
              +str(np.max(np.abs(m1+eh(m2)))))
    print("CHECKED that the Hamiltonian has electron-hole symmetry")


def check_dict(mf):
    """Check a dictionary with hopping, like the one used for mean field"""
    for key in mf:
        key2 = tuple([-i for i in key])
        m1 = mf[key]
        m2 = mf[key2]
        if not equal(m1,np.conjugate(m2).T):
            print(key,key2)
            print("First")
            print(m1)
            print("Second")
            print(m2)




# Hilbert-space guards
#
# A routine that needs the spin or the electron-hole (Nambu) degree of
# freedom used to raise its own hand-written ValueError, so the same
# requirement was worded ~30 different ways across the package and a new
# routine had to reinvent both the wording and the remedy. These three
# functions are the single home for it: each takes the Hamiltonian and a
# noun phrase naming what needs the degree of freedom, and supplies the
# fixed tail that tells the user how to get it.
#
#     require_spin(h,"an exchange field")
#     -> "an exchange field needs a spinful Hamiltonian; call
#         h.turn_spinful() first, or build it with
#         g.get_hamiltonian(has_spin=True)"


def require_spin(h,what):
  """Raise unless h carries the spin degree of freedom

  what: the noun phrase the message opens with, naming the routine or
      quantity that needs spin (e.g. "the Kane-Mele coupling").
  """
  if not h.has_spin:
      raise ValueError(str(what)+" needs a spinful Hamiltonian; call "
        "h.turn_spinful() first, or build it with "
        "g.get_hamiltonian(has_spin=True)")


def require_nambu(h,what):
  """Raise unless h carries the electron-hole (Nambu) degree of freedom

  what: the noun phrase the message opens with, naming the routine or
      quantity that needs the Nambu spinor (e.g. "the d-vector").
  """
  if not h.has_eh:
      raise ValueError(str(what)+" needs a Nambu Hamiltonian, with the "
        "electron-hole degree of freedom; call h.setup_nambu_spinor() "
        "first")


def require_sublattice(h,what):
  """Raise unless the geometry of h is labeled with a sublattice

  Sublattice-resolved quantities are silently meaningless without it --
  several callers used to return None or an array of zeros instead of
  saying so.
  """
  g = h.geometry if hasattr(h,"geometry") else h # accept a Geometry too
  if not g.has_sublattice:
      raise ValueError(str(what)+" needs a geometry with a sublattice, and "
        "this one has none. Build a cell that fits the modulation and label "
        "it -- g = g.get_supercell(2) followed by g.get_sublattice(), which "
        "two-colors the lattice -- or pass an explicit input instead")
