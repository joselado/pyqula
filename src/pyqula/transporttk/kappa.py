# compute the kappa parameter of a heterostructure
import numpy as np
from ..parallel import pcall

def get_single(HT=None,c=1.0,energies=[0.0]):
    """Get a single conductance"""
    HT.scale_rc = c # scaling
    return np.array([HT.didv(energy=e) for e in energies]) # loop over Ts


def get_conductances(T=1e-2,**kwargs):
    """Compute Kappa by doing a log-log plot"""
    cref = T
    ts = np.exp(np.linspace(np.log(cref*0.9),np.log(cref*1.1),4)) # hoppings
#    ts = [cref*0.99,cref*1.01]
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


def get_kappa_ratio(HT,delta=1e-10,**kwargs):
    ks1 = get_kappa(HT=generate_HT(HT,delta=delta,SC=True),**kwargs)
    ks2 = get_kappa(HT=generate_HT(HT,delta=delta,SC=False),**kwargs)
    return ks1/ks2


def generate_HT(ht,SC=True,delta=1e-10):
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
    hto = build(Hr,Hl) # create a new heterostructure
    hto.delta = delta
    return hto





