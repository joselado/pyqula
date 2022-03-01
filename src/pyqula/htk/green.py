from ..greentk.selfenergy import bloch_selfenergy


def get_gf(H,energy=0.0,delta=1e-5,
           gtype="bulk",
           mode="adaptive",
           **kwargs):
    """Generate the specific Green's function"""
    if H.dimensionality>0:
        gf = bloch_selfenergy(H,energy=energy,
                                         mode=mode,
                                         gtype=gtype,
                                         delta=delta)[0]
        return gf
    elif H.dimensionality==0: # zero dimensional
        return H.get_gk_gen(delta=delta)(e=energy) 
    else: raise

