# different functions specialized for graphene

def add_fwave(h,v,sign=1):
    """Add fwave superconductivity"""
    h0 = h.copy()
    from ..superconductivity import hopping2deltaud
    h0 = h0*0. 
    if sign==1: h0.add_antihaldane(v) 
    elif sign==-1: h0.add_haldane(v) 
    h = hopping2deltaud(h,h0*1j)
    return h


