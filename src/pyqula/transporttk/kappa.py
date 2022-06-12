# compute the kappa parameter of a heterostructure
import numpy as np
from ..parallel import pcall

def get_single(HT=None,c=1.0,energies=[0.0],**kwargs):
    """Get a single conductance"""
    HT.set_coupling(c) # scaling
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

def get_kappa(energy=0.0,**kwargs):
    ts,Gs = get_conductances(energies=[energy],**kwargs)
    ks = []
    for g in Gs.T: # loop over energies
        k = get_power(ts,g)
        ks.append(k)
    return np.array(ks)[0] # return kappa


def get_kappa_ratio(HT,**kwargs):
    ks1 = get_kappa(HT=generate_HT(HT,SC=True,**kwargs),**kwargs)
    ks2 = get_kappa(HT=generate_HT(HT,SC=False,**kwargs),**kwargs)
    return ks1/ks2


def generate_HT(ht,SC=True,**kwargs):
    """Given a heterostructure, generate a new one to compute kappa"""
    # this is a workaround
    from ..heterostructures import Heterostructure
    from .localprobe import LocalProbe
    from ..embeddingtk.kappa import DualLocalProbe
    def f(h):
        h = h.copy()
        if not SC: # remove the SC order
            h.remove_nambu()
            h.setup_nambu_spinor()
        return h
    if type(ht)==Heterostructure: # heterostructure type
        from ..heterostructures import build
        Hr = f(ht.Hr)
        Hl = f(ht.Hl)
        hto = build(Hl,Hr) # create a new heterostructure
        hto.delta = ht.delta
        return hto
    elif type(ht)==LocalProbe: # Localprobe type
        out = ht.copy() # make a copy
        out.H = f(out.H)
        out.lead = f(out.lead)
        return out
    elif type(ht)==DualLocalProbe: # Dual Localprobe object
        if SC: return ht.SC_LP
        else: return ht.normal_LP
    else: raise


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

