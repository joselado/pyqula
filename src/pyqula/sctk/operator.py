

def real_singlet(h):
    """Given a Hamiltonian, return an operator that computes the
    real part of the spin singlet pairing"""
    # add_swave promotes its argument into Nambu space, so without this
    # check the operator came out bigger than the Hilbert space it is meant
    # to act on, and only failed later inside a raw numpy matmul
    if not h.has_eh:
        raise ValueError("a pairing operator needs the electron-hole "
          +"(Nambu) degree of freedom; call h.setup_nambu_spinor() first")
    if not h.check_mode("spinful_nambu"):
        raise NotImplementedError("the 'singlet' operator is the spin "
          +"singlet component of the pairing, in the spin x electron-hole "
          +"basis, so it is only defined for a spinful Nambu Hamiltonian; "
          +"this one is spinless Nambu. Use h.extract('swave') or "
          +"sctk.spinless for the spinless case")
    op = h.copy()*0. # initialize
    op.add_swave(1.0) # add pairing
    from ..operators import Operator
    return Operator(op.intra) # return
