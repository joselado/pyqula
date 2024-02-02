# library to compute the LDOS


def get_ldos(self,mode="diagonalization",**kwargs):
    # only the ED mode was implemented
    from ..ldos import get_ldos_general
    if mode !="diagonalization": # others should be implemented
        print(mode, "not implemented")
    return get_ldos_general(self,mode=mode,
            non_hermitian=True,**kwargs)



