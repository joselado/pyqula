# library to compute the LDOS


def get_ldos(self,mode="",**kwargs):
    # only the ED mode was implemented
    return get_dos_general(self,mode="diagonalization",**kwargs)


