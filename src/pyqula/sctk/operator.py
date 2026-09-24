
from ..check import require_nambu

def real_singlet(h):
    """Given a Hamiltonian, return an operator that computes the
    real part of the spin singlet pairing"""
    # add_swave promotes its argument into Nambu space, so without this
    # check the operator came out bigger than the Hilbert space it is meant
    # to act on, and only failed later inside a raw numpy matmul
    require_nambu(h,"a pairing operator")
    op = h.copy()*0. # initialize
    op.add_swave(1.0) # add pairing
    from ..operators import Operator
    return Operator(op.intra) # return
