from ..green import bloch_selfenergy


def get_gf(H,energy=0.0,delta=1e-5,
           gtype="bulk",
           mode="adaptive",
           **kwargs):
    """Generate the specific Green's function"""
    if gtype=="bulk":
        gf = bloch_selfenergy(H,energy=energy,
                                         mode=mode,
                                         delta=delta)[0]
        return gf
    elif gtype=="surface":
        gf = bloch_selfenergy(H,energy=energy,
                                         mode=mode,
                                         delta=delta)[1]
        return gf
    else: raise # not implemented

