import numpy as np

# Return topological object in a sector of an operator #

def get_berry_curvature_operator_sector(H,operator=None,sector=1.0,
        **kwargs):
    """Return the Berry curvature in one sector"""
    from .occstates import occ_states_sector_generator
    H = H.copy() # make a copy
    H.os_gen = occ_states_sector_generator(H,operator=operator,
            sector=sector)
    return H.get_berry_curvature(**kwargs)


def get_chern_operator_sector(H,operator=None,sector=1.0,
        **kwargs):
    """Return the Chern number in one operator sector"""
    from .occstates import occ_states_sector_generator
    H = H.copy() # make a copy
    H.os_gen = occ_states_sector_generator(H,operator=operator,
            sector=sector)
    from ..topology import mesh_chern
    return mesh_chern(H,**kwargs)


