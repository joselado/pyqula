# compute the kappa parameter of a heterostructure
from contextlib import contextmanager
import numpy as np
from ..parallel import pcall

def get_single(HT=None,c=1.0,energies=[0.0],**kwargs):
    """Get a single conductance"""
    HT.set_coupling(c) # scaling
    return np.array([HT.didv(energy=e,**kwargs) for e in energies]) # loop over Ts


@contextmanager
def _selfenergy_cache_scope(ht):
    """Enable LocalProbe self-energy caching (LocalProbe.reuse_selfenergy)
    for the duration of the block, restoring whatever state was there
    before on exit. Exact, not an approximation: neither selfenergy
    LocalProbe.get_selfenergy returns depends on the probe-sample coupling
    ht.T (get_central_gmatrix only ever scales the off-diagonal coupling
    block by T), so sharing them across get_conductances' two-coupling
    sweep at fixed energy changes nothing about the result -- only how
    many times the expensive Sancho-Rubio/sample-GF selfenergy solve gets
    redone (2x -> 1x per coupling sweep). `ht` types without this cache
    (e.g. Heterostructure) silently get a no-op here."""
    if not hasattr(ht,"reuse_selfenergy"):
        yield
        return
    prev_flag,prev_cache = ht.reuse_selfenergy,ht._selfenergy_cache
    ht.reuse_selfenergy = True
    ht._selfenergy_cache = {}
    try:
        yield
    finally:
        ht.reuse_selfenergy = prev_flag
        ht._selfenergy_cache = prev_cache


def get_conductances(T=1e-2,**kwargs):
    """Compute Kappa by doing a log-log plot"""
    cref = T
    ts = np.exp(np.linspace(np.log(cref*0.9),np.log(cref*1.1),2)) # hoppings
#    ts = [cref*0.9,cref*1.1]
    with _selfenergy_cache_scope(kwargs.get("HT")):
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


def _with_shared_selfenergy(ht,kwargs):
    """Return kwargs with a shared AAA self-energy interpolant added,
    covering this call's single energy (get_kappa's `energy` kwarg,
    defaulting to 0.0), if `ht` is Floquet-Keldysh-eligible and the
    caller hasn't already supplied their own selfenergy_qtci -- otherwise
    kwargs unchanged. Building it once here, instead of letting
    get_conductances's two-coupling probe (get_single, one didv() call
    per coupling) each independently build their own default fit, is the
    same sharing _shared_selfenergy_for_branch already does for the
    finite-temperature kappa path below, applied to the zero-temperature
    one -- a smaller win (only 2 calls share one build here, versus that
    path's whole coupling/energy/thermal-quadrature sweep) but the same
    already-proven, zero-risk pattern."""
    if "selfenergy_qtci" in kwargs: return kwargs
    from .didv import _both_leads_superconducting
    if not _both_leads_superconducting(ht): return kwargs
    from ..keldyshtk.current import build_shared_selfenergy
    energy = kwargs.get("energy",0.0)
    nmax_max = kwargs.get("nmax_max",40)
    shared = build_shared_selfenergy(ht,abs(energy),nmax_max=nmax_max,
                        delta=kwargs.get("delta"),dv=kwargs.get("dv"))
    if shared is None: return kwargs
    return dict(kwargs,selfenergy_qtci=shared)


def get_kappa_ratio(HT,**kwargs):
    # SC branch only: the normal branch below never routes through Keldysh
    # (generate_HT(...,SC=False) strips pairing), so it has no self-energy
    # interpolant to share and must keep using the caller's own kwargs
    # unchanged -- injecting the SC branch's interpolant there would hand
    # it the wrong leads' self-energy.
    ht_sc = generate_HT(HT,SC=True,**kwargs)
    ht_normal = generate_HT(HT,SC=False,**kwargs)
    # Zero-temperature, 1D LocalProbe fast path: kappa here is
    # d(log G)/d(log t), which get_kappa/get_power below estimate with a
    # 2-point secant (2 extra didv/selfenergy solves per branch). When
    # both branches are LocalProbe objects with a non-superconducting
    # probe (transporttk.kappa_jax.applicable), that derivative can be
    # computed exactly with jax.grad instead, from a single selfenergy
    # solve per branch -- see kappa_jax's module docstring for the
    # measured ~3.7x speedup and accuracy comparison. get_kappa_ratio_jax
    # returns None (jax unavailable, wrong case, or a solve failure) for
    # anything outside that scope, in which case the code below runs
    # exactly as before.
    from .kappa_jax import get_kappa_ratio_jax
    energy = kwargs.get("energy",0.0)
    T = kwargs.get("T",1e-2)
    fast = get_kappa_ratio_jax(ht_sc,ht_normal,energy=energy,T=T)
    if fast is not None: return fast
    ks1 = get_kappa(HT=ht_sc,**_with_shared_selfenergy(ht_sc,kwargs))
    ks2 = get_kappa(HT=ht_normal,**kwargs)
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


def _shared_selfenergy_for_branch(ht,energies,temp,nmax_max=40,delta=None,dv=None,**kwargs):
    """Thin wrapper around keldyshtk.current.build_shared_selfenergy,
    sized to cover every quasienergy the finite-temperature thermal
    quadrature (transporttk.thermaldidv.finite_T_didv, window
    +-thermaldidv.THERMAL_WINDOW*temp) could visit for any energy in
    `energies` -- so it can be built ONCE per SC/normal branch and shared
    across every (coupling, energy, thermal-quadrature-node) combination
    get_kappa_finite_temperature_energies evaluates for that branch,
    instead of each one rebuilding its own call-local fit. That per-call
    rebuilding is what made the un-shared, finite-temperature keldysh path
    this replaces prohibitively slow: measured to not even finish within
    several minutes at settings loose enough that the underlying
    dc_current calls hadn't converged either.

    Self-energy is purely a lead property: it does not depend on the
    inter-lead coupling (get_conductances_finite_temp scans two couplings
    via HT.set_coupling) nor on which entry of `energies` is being
    evaluated, so one interpolant sized to cover the whole sweep's energy
    range is valid for the entire calculation, not just one point of it.

    See build_shared_selfenergy's own docstring for the full None-return
    contract (not both leads superconducting, or the AAA fit didn't
    converge within budget) and for why an explicit `dv` must be honored
    here too (SelfenergyAAA performs no domain check and would silently
    extrapolate for a call whose voltage+-dv pushes past the fitted
    window)."""
    from .thermaldidv import THERMAL_WINDOW
    from ..keldyshtk.current import build_shared_selfenergy
    emax = max(abs(e) for e in energies) if len(energies) else 0.
    vmax = emax + THERMAL_WINDOW*temp
    return build_shared_selfenergy(ht,vmax,nmax_max=nmax_max,delta=delta,dv=dv)


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
        # Only pass selfenergy_qtci when there is a real, converged
        # interpolant to share. Passing selfenergy_qtci=None explicitly
        # (rather than omitting the key) would make keldysh_didv skip its
        # own "if 'selfenergy_qtci' not in kwargs" auto-build entirely
        # (the key is present, just with value None), disabling even its
        # pre-existing per-call sharing between one didv() call's own
        # Ip/Im finite-difference pair and forcing two independent builds
        # instead of one -- strictly worse than not touching this kwarg
        # at all, which is what the non-superconducting branch (and a
        # superconducting branch whose build didn't converge) should get.
        extra = {"selfenergy_qtci": shared} if shared is not None else {}
        # dict.update, not **extra,**kwargs: if the caller already passed
        # their own selfenergy_qtci in kwargs, unpacking both would raise
        # "got multiple values for keyword argument" instead of letting
        # the freshly-built, branch-specific shared interpolant win
        call_kwargs = dict(kwargs)
        call_kwargs.update(extra)
        ts,Gs = get_conductances_finite_temp(
            HT=ht,energies=energies,temp=temp,**call_kwargs)
        return np.array([get_power(ts,g) for g in Gs.T])
    ks1 = branch_kappas(True)
    ks2 = branch_kappas(False)
    return ks1/ks2

