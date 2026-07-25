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


#### Finite-temperature kappa: same SC/normal power-law-ratio idea as
#### get_kappa_ratio above, but each conductance is thermally averaged
#### (HT.didv(temp=...), see transporttk.thermaldidv.finite_T_didv)
#### instead of computed at T=0. This used to be an unfinished stub: the
#### function below was defined twice under the same name (the second
#### definition silently shadowed the first, so calling it recursed into
#### itself forever) and the first definition called an undefined
#### get_conductances_finite_temp -- both are fixed here.

def get_conductances_finite_temp(T=1e-2,temp=1e-2,**kwargs):
    """Finite-temperature analog of get_conductances: same two-coupling
    log-log sampling used to extract the kappa power-law exponent, but
    each conductance is HT.didv(temp=temp,...) (thermally averaged)
    rather than the T=0 conductance get_conductances uses."""
    cref = T
    ts = np.exp(np.linspace(np.log(cref*0.9),np.log(cref*1.1),2)) # hoppings
    Gs = np.array([get_single(c=t,temp=temp,**kwargs) for t in ts]) # compute conductance
    return ts,Gs


def _shared_selfenergy_for_branch(ht,energies,temp,nmax_max=40,delta=None,**kwargs):
    """Build one aaatk.selfenergy_aaa.SelfenergyAAA lead self-energy
    interpolant per lead (see keldyshtk.current.build_selfenergy_aaa),
    covering every quasienergy the finite-temperature thermal quadrature
    (transporttk.thermaldidv.finite_T_didv, window
    +-thermaldidv.THERMAL_WINDOW*temp) could visit for any energy in
    `energies` -- so it can be built ONCE per SC/normal branch and shared
    across every (coupling, energy, thermal-quadrature-node) combination
    get_kappa_finite_temperature_energies evaluates for that branch,
    instead of keldysh_didv rebuilding its own call-local fit at every
    single one of them (its use_aaa=True default only shares a fit within
    one didv() call's own Ip/Im finite-difference pair, not across the
    many outer calls a thermal integral or a kappa coupling/energy sweep
    makes). That per-call rebuilding is what makes the un-shared,
    finite-temperature keldysh path this replaces prohibitively slow:
    measured to not even finish within several minutes at settings loose
    enough that the underlying dc_current calls hadn't converged either.

    Self-energy is purely a lead property: it does not depend on the
    inter-lead coupling (get_conductances_finite_temp scans two couplings
    via HT.set_coupling) nor on which entry of `energies` is being
    evaluated, so one interpolant sized to cover the whole sweep's energy
    range is valid for the entire calculation, not just one point of it.

    Returns None if this branch's leads are not both superconducting
    (didv's "auto" method then picks "smatrix", already cheap -- no
    self-energy interpolant is needed there, see transporttk.didv.didv's
    docstring) or if the AAA fit doesn't converge within its default
    build budget -- callers get back the ordinary per-call default
    self-energies in that case (build_selfenergy_aaa/dc_current's own
    fallback contract), never a wrong answer."""
    from .didv import _both_leads_superconducting
    if not _both_leads_superconducting(ht):
        return None
    from ..keldyshtk.current import build_selfenergy_aaa
    from .thermaldidv import THERMAL_WINDOW
    if delta is None: delta = ht.delta
    emax = max(abs(e) for e in energies) if len(energies) else 0.
    vmax = emax + THERMAL_WINDOW*temp
    vmax += max(vmax*1e-2,1e-3) # keldysh_didv's own default finite-difference dv
    shared = build_selfenergy_aaa(ht,vmax,nmax_max,delta=delta)
    if not all(s.converged for s in shared.values()):
        return None
    return shared


def get_kappa_finite_temperature_energies(HT,energies=[0.0],temp=1e-2,**kwargs):
    """Finite-temperature kappa over an array of energies: the same
    SC/normal power-law ratio as get_kappa_ratio, but every conductance
    entering the fit is thermally averaged at temperature `temp` instead
    of computed at T=0. For whichever branch (SC or normal) turns out to
    actually be superconducting -- and therefore routes through the
    expensive Floquet-Keldysh dI/dV rather than the cheap smatrix formula,
    see transporttk.didv.didv's docstring -- one AAA self-energy
    interpolant is built once up front and shared across that branch's
    whole sweep (_shared_selfenergy_for_branch) instead of being rebuilt
    from scratch at every coupling/energy/thermal-quadrature-node
    combination."""
    def branch_kappas(sc):
        ht = generate_HT(HT,SC=sc,**kwargs)
        shared = _shared_selfenergy_for_branch(ht,energies,temp,**kwargs) if sc else None
        ts,Gs = get_conductances_finite_temp(
            HT=ht,energies=energies,temp=temp,selfenergy_qtci=shared,**kwargs)
        return np.array([get_power(ts,g) for g in Gs.T])
    ks1 = branch_kappas(True)
    ks2 = branch_kappas(False)
    return ks1/ks2

