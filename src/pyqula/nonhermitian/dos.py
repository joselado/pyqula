
# woorkaround for non Hermitian Hamiltonians

from ..dos import get_dos_general

def get_dos(self,mode="",**kwargs):
    # overwrite the method to use ED
    return get_dos_general(self,mode="ED",**kwargs)

