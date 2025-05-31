from ..operators import Operator

def get_angular_momenta(H,**kwargs):
    """Return the angular momenta operator"""
    op1 = H.copy()
    op2 = H.copy()
    dB = 1e-4 # infinitesimal
    op1.add_peierls(dB)
    op2.add_peierls(-dB)
    op = (op2-op1)*(1./dB)
    op = Operator(op)
    return op # return operator

