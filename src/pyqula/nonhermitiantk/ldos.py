# library to compute the LDOS


def get_ldos(self,mode="diagonalization",**kwargs):
    # only the ED mode is implemented
    from ..ldos import get_ldos_general
    if mode !="diagonalization": # others should be implemented
        # this used to print and then forward the unsupported mode anyway,
        # so the caller got the diagonalization answer labelled as
        # something else
        raise NotImplementedError("the non-Hermitian LDOS is only "
                "implemented for mode='diagonalization', got mode='"
                +str(mode)+"'")
    return get_ldos_general(self,mode=mode,
            non_hermitian=True,**kwargs)



