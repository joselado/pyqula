

def real_singlet(h):
    """Given a Hamiltonian, return an operator that computes the
    real part of the spin singlet pairing"""
    op = h.copy()*0. # initialize
    op.add_swave(1.0) # add pairing
    from ..operators import Operator
    return Operator(op.intra) # return

