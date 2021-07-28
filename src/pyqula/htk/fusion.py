import numpy as np


def hamiltonian_fusion(h1,h2):
    """Given two Hamiltonians, fusion them"""
    go = h1.geometry + h2.geometry # sum the geometries
    h1 = h1.get_multicell() # to multicell form
    ho = h1.copy() # copy Hamiltonian
    ho.geometry = go # update geometry
    d1 = h1.get_dict() # get dictionary
    d2 = h2.get_dict() # get dictionary
    from ..multihopping import direct_sum_hopping_dict
    do = direct_sum_hopping_dict(d1,d2)
    from ..multihopping import MultiHopping
    mho = MultiHopping(do)
    ho.set_multihopping(mho)
    return ho # return Hamiltonian



