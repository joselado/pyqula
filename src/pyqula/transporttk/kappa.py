# compute the kappa parameter of a heterostructure
import numpy as np
from ..parallel import pcall

def get_single(HT=None,c=1.0,energies=[0.0],**kwargs):
    """Get a single conductance"""
    HT.scale_rc = c # scaling
    return np.array([HT.didv(energy=e,**kwargs) for e in energies]) # loop over Ts


def get_conductances(T=1e-2,**kwargs):
    """Compute Kappa by doing a log-log plot"""
    cref = T
    ts = np.exp(np.linspace(np.log(cref*0.9),np.log(cref*1.1),2)) # hoppings
#    ts = [cref*0.9,cref*1.1]
    Gs = np.array([get_single(c=t,**kwargs) for t in ts]) # compute conductance
    return ts,Gs

def get_power(ts,gs,delta=1e-8):
    """Given hoppings and conductances, extract the power"""
#    ts2 = ts[gs>delta]
#    gs2 = gs[gs>delta]
    p = np.polyfit(np.log(ts),np.log(gs),1)
    k = p[0]
    return k

def get_kappa(**kwargs):
    ts,Gs = get_conductances(**kwargs)
    ks = []
    for g in Gs.T: # loop over energies
        k = get_power(ts,g)
        ks.append(k)
    return np.array(ks) # return kappa


def get_kappa_ratio(HT,**kwargs):
    ks1 = get_kappa(HT=generate_HT(HT,SC=True,**kwargs),**kwargs)
    ks2 = get_kappa(HT=generate_HT(HT,SC=False,**kwargs),**kwargs)
    return ks1/ks2


def generate_HT(ht,SC=True,temperature=0.,**kwargs):
    """Given a heterostructure, generate a new one to compute kappa"""
    def f(h):
        h = h.copy()
        if not SC: # remove the SC order
            h.remove_nambu()
            h.setup_nambu_spinor()
        return h
    from ..heterostructures import build
    Hr = f(ht.Hr)
    Hl = f(ht.Hl)
    hto = build(Hl,Hr) # create a new heterostructure
    hto.delta = ht.delta
#    hto.extra_delta_right = temperature
#    hto.extra_delta_left = temperature
    return hto


#### These are workrounds for more efficient finite temperature calculations ##
## Not yet finished! ##

def get_kappa_finite_temperature_energies(**kwargs):
    """Compute kappa using temperature convolution"""
    ts,Gs = get_conductances_finite_temp(**kwargs) # G at T=0
    ks = []
    for g in Gs.T: # loop over energies
        k = get_power(ts,g)
        ks.append(k)
    return np.array(ks) # return kappa


def get_kappa_finite_temperature_energies(HT,**kwargs):
    ks1 = get_kappa_finite_temperature_energies(
                     HT=generate_HT(HT,SC=True,**kwargs),**kwargs)
    ks2 = get_kappa_finite_temperature_energies(
                     HT=generate_HT(HT,SC=False,**kwargs),**kwargs)
    return ks1/ks2

