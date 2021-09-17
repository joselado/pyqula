

def surface_dos(ht,lead="r",**kwargs):
    """Return the surface DOS"""
    if lead=="r": H = ht.Hr
    else: H = ht.Hl
    from ..dos import surface_dos
    return surface_dos(H,**kwargs)

