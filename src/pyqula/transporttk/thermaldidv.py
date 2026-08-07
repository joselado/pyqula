import numpy as np
from scipy.integrate import quad

thermalmode = "adaptive" # thermal mode

THERMAL_WINDOW = 20 # max |energy-energy0|/temp the thermal quad below integrates
                     # over; exposed as a module constant (rather than a
                     # value local to finite_T_didv) so other callers that
                     # need to know this function's effective energy range
                     # ahead of time (e.g. kappa.py's finite-temperature
                     # path, which builds a self-energy interpolant sized
                     # to cover it) don't have to duplicate the magic number.

def finite_T_didv(self,temp,energy=0.0,keldysh_thermal_mode="direct",**kwargs):
    """Finite temperature dIdV.

    `keldysh_thermal_mode` picks how a Floquet-Keldysh-eligible junction's
    (see transporttk.didv._both_leads_superconducting) finite-temperature
    dI/dV is obtained -- it has no effect otherwise (a non-Keldysh
    junction always uses the "convolution" approach below, its only exact
    formalism-agnostic option):

    "direct" (default) evaluates keldyshtk.current.dc_current's own native
    `temperature` parameter, which broadens EACH Floquet sideband's own
    occupation by `temp` independently (see dc_current's `_fermi_scalar`/
    `_assemble_chain_jit`) -- one central-bias-difference dI/dV
    (transporttk.didv.keldysh_didv, 2 dc_current calls total) instead of
    the ~150-300 independent T=0 dc_current-pair evaluations "convolution"
    below needs (147 measured for one "adaptive"-mode call at temp=0.02).
    IMPORTANT: this is NOT simply a faster way to compute what
    "convolution" computes -- see documentation/
    keldysh_sideband_decimation_plan.md's "direct finite-T Keldysh
    evaluation" entry for why the two are structurally different
    quantities (convolution smears BIAS, which shifts every Floquet
    sideband by an n-dependent amount; direct broadens each sideband's own
    occupation by a fixed temp) and only have to agree as temp->0, not
    away from that limit. Validated against an independent
    (non-Floquet) finite-temperature Landauer reference for a
    normal-normal reduction in tests/keldysh/
    test_normal_junction_finite_temperature.py before this became the
    default.

    "convolution" is the original approach (below): evaluate zero_T_didv
    at many energies within +-THERMAL_WINDOW*temp of `energy` -- all of
    them on the SAME junction, so if it is Floquet-Keldysh-eligible AND
    the caller explicitly opted into selfenergy_method="aaa" (see
    dc_current's own docstring for why that's not the default -- an
    unresolved accuracy gap), one shared AAA self-energy interpolant
    (keldyshtk.current.build_shared_selfenergy), sized to cover the whole
    window up front, is built once here and reused by every one of those
    evaluations instead of each independently building (and discarding)
    its own -- the same sharing kappa.py's _shared_selfenergy_for_branch
    already does for its own outer coupling/energy sweep, applied here to
    the window a single finite_T_didv call visits internally, with or
    without any outer sweep at all. Skipped if the caller already passed
    selfenergy_qtci explicitly, or didn't ask for
    selfenergy_method="aaa". This is also the only mode used for a
    non-Keldysh (e.g. plain "smatrix") junction, where it is the exact,
    standard thermal-broadening-of-a-T=0-characteristic-curve formula
    (transmission is temperature-independent there, unlike a Floquet
    sideband ladder)."""
    from .didv import zero_T_didv, _both_leads_superconducting, keldysh_didv
    if keldysh_thermal_mode not in ("direct", "convolution"):
        raise ValueError("keldysh_thermal_mode must be 'direct' or "
                          f"'convolution', got {keldysh_thermal_mode!r}")
    if keldysh_thermal_mode == "direct" and _both_leads_superconducting(self):
        delta = kwargs.pop("delta", None)
        if delta is None: delta = self.delta
        dv = kwargs.pop("dv", None)
        if dv is None: dv = max(abs(energy)*1e-2, 1e-3)
        # keldysh_didv's default dv is tuned for the T=0 finite difference,
        # where there is no thermal scale to resolve; at finite temp, a dv
        # comparable to or larger than temp would smooth away exactly the
        # thermal structure this mode exists to capture (e.g. temp=1e-3,
        # voltage=1.0 gives the untouched default dv=1e-2, 10x the thermal
        # width) -- clamp it well below temp instead.
        if temp > 0: dv = min(dv, 0.1*temp)
        return keldysh_didv(self, voltage=energy, delta=delta, dv=dv,
                             temperature=temp, **kwargs)

    if ("selfenergy_qtci" not in kwargs and kwargs.get("selfenergy_method") == "aaa"
            and _both_leads_superconducting(self)):
        from ..keldyshtk.current import build_shared_selfenergy
        nmax_max = kwargs.get("nmax_max", 40)
        shared = build_shared_selfenergy(self, abs(energy)+THERMAL_WINDOW*temp,
                nmax_max=nmax_max, delta=kwargs.get("delta"), dv=kwargs.get("dv"))
        if shared is not None:
            kwargs = dict(kwargs, selfenergy_qtci=shared)
    if thermalmode=="adaptive":
        from .fermidirac import fermidirac as FD
        dt = THERMAL_WINDOW # max T range
        de = temp # energy difference to compute the derivative
        ### Use simpson integration
        def f(e):
            out = zero_T_didv(self,energy=energy+e,**kwargs)
            out *= FD(e-de,temp=temp) - FD(e+de,temp=temp)
            return out/de
        from ..integration import peak_integrate
        return quad(f,-dt*temp,dt*temp,epsrel=1e-4,limit=60)[0]/2.
    elif thermalmode=="pm": # poor's man mode
        from .fermidirac import dFDdT
        nT = 45
        Ts = np.linspace(-4*temp,4*temp,nT)
        def f(e):
            return zero_T_didv(self,energy=energy+e,**kwargs)
        out = [f(e)*dFDdT(e,temp=temp)*temp for e in Ts]
        norm = np.sum(dFDdT(Ts,temp=temp)*temp)
        out = np.trapezoid(out)/norm
        return out
    else: raise


