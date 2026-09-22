import numpy as np
from .. import algebra
from ..green import green_renormalization
from .. import green
from copy import deepcopy
from ..htk.mode import make_compatible

delta_smatrix = 1e-12
dagger = algebra.dagger
gfmode = "adaptive"

# library to perform transport calculations using a local probe


class LocalProbe():
    def __init__(self,h,lead=None,delta=1e-6,i=0,T=1.0,**kwargs):
        h = h.get_dense() # dense Hamiltonian
        self.H = h.copy() # store Hamiltonian
        # Precompute the non-multicell form once, when valid (nearest-
        # neighbor or shorter hopping only -- the same precondition
        # bloch_selfenergy itself uses before ever calling
        # get_no_multicell()), instead of leaving every later self-energy
        # call (bloch_selfenergy's per-energy sideband sweep in
        # keldyshtk/current.py's Floquet-Keldysh dc_current) to redo that
        # (expensive, deepcopy-based) conversion from scratch against an
        # unchanging Hamiltonian. Mirrors the identical precomputation
        # already done for the probe lead below. For longer-range hopping,
        # bloch_selfenergy takes a different code path that never calls
        # get_no_multicell(), so there's nothing to precompute there.
        from ..htk.kchain import detect_longest_hopping
        if self.H.is_multicell and detect_longest_hopping(self.H)<=1:
            self.H = self.H.get_no_multicell()
        self.has_eh = self.H.has_eh # electron-hole
        self.dimensionality = 1 # probe-to-single-site coupling, always 1D
        self.delta = delta
        self.mode = "bulk"
        self.reuse_gf = False # reuse the Green's function
        self.gf = None
        # Neither selfenergy (probe lead=0, sample lead=1) depends on the
        # probe-sample coupling self.T -- see get_central_gmatrix, where T
        # only scales the off-diagonal coupling block -- so callers that
        # sweep T at fixed energy (transporttk.kappa's get_conductances)
        # can safely cache them across that sweep. Off by default so every
        # other call site keeps computing them fresh, as before.
        self.reuse_selfenergy = False
        self._selfenergy_cache = {}
        self.bulk_delta = delta
        self.frozen_lead = True
        self.i = i # this site
        if lead is None:
            from ..geometry import chain 
            lead = chain().get_hamiltonian(has_spin=False) # create a chain
        lead = make_compatible(lead,self.H) # make them compatible
        lead = lead.get_no_multicell() # no multicell
        self.lead = lead.copy() # store
        self.get_eh_sector = self.lead.get_eh_sector # if it has electron-hole
        self.T = T # transparency
    def get_selfenergy(self,energy,lead=0,**kwargs):
        """Return the selfenergies"""
        if self.reuse_selfenergy:
            # keyed on every kwarg that can change the result (delta,
            # numba) besides energy/lead -- not just (energy,lead) -- so a
            # cache scope spanning calls with different delta/numba can't
            # silently return a stale selfenergy solved with the wrong one.
            # lead=1 (system) additionally depends on self.i (the probe
            # site, see local_selfenergy below) while lead=0 (probe) does
            # not -- so only lead=1's key includes it. This lets callers
            # that sweep self.i at fixed energy (e.g. embeddingtk.didv's
            # per-site dI/dV map) safely reuse lead=0's selfenergy across
            # every site instead of recomputing the same Sancho-Rubio
            # solve each time, without lead=1 entries from different sites
            # colliding in the cache.
            key = (energy,lead,kwargs.get("delta"),kwargs.get("numba"))
            if lead==1: key = key+(self.i,)
            if key in self._selfenergy_cache: return self._selfenergy_cache[key]
        if lead==0: # use the probe
            out = lead_selfenergy(self,energy=energy,**kwargs)
        elif lead==1: # use the system
            g = generate_gf(self,energy=energy,
                               **kwargs) # generate the Green's function
            out = local_selfenergy(self.H,g,i=self.i,
                                energy=energy,**kwargs)
        else:
            raise ValueError("a local probe has two leads, so lead must be 0 "
                    "or 1")
        if self.reuse_selfenergy: self._selfenergy_cache[key] = out
        return out
    def get_selfenergy_batch(self,energies,lead=0,**kwargs):
        """Batched twin of get_selfenergy: one lead, many energies at
        once, returned as one (len(energies),dim,dim) array. The
        counterpart of transporttk.selfenergy.get_selfenergy_batch for a
        Heterostructure, and the reason keldyshtk.current's
        `_prefetch_selfenergies_batch` stops falling back to per-energy
        solves here: both of a probe's selfenergies come down to a
        Sancho-Rubio decimation on one fixed (intra,inter) pair, which is
        independent across energies and so runs as a single numba
        prange-parallel call instead of tens of thousands of Python-level
        ones per Floquet dI/dV point.

        `self._selfenergy_cache` is deliberately not consulted or filled,
        exactly as the Heterostructure version does not: the caller that
        needs this batching keeps its own energy-keyed cache around the
        whole sideband sweep."""
        energies = np.asarray(energies)
        if lead==0: # use the probe
            return lead_selfenergy_batch(self,energies,**kwargs)
        elif lead==1: # use the system
            g = generate_gf_batch(self,energies,**kwargs)
            return local_selfenergy_batch(self.H,g,i=self.i,
                                energies=energies,**kwargs)
        else:
            raise ValueError("a local probe has two leads, so lead must be 0 "
                    "or 1")
    def get_central_gmatrix(self,**kwargs):
        return get_central_gmatrix(self,**kwargs)
    def with_delta(self,delta):
        """Return a copy of this probe whose broadening is `delta`, as if
        it had been built with LocalProbe(...,delta=delta) -- __init__
        sets both the probe's own delta and the bulk_delta of the sample
        Green's function from its single delta argument, and both of them
        change the answer, so both are set here. This is how a `delta=`
        keyword given to a single didv/get_smatrix call is made to mean
        exactly what the constructor argument means."""
        from copy import copy
        out = copy(self) # shallow, the Hamiltonians are shared
        out.delta = delta # probe selfenergy and central Green's function
        out.bulk_delta = delta # Green's function of the sample
        # the cache is keyed on the (clamped) delta that get_smatrix hands
        # down, not on this one, so the copy must not share its entries,
        # and a reused Green's function was solved at the old bulk_delta
        out._selfenergy_cache = {}
        out.gf = None
        return out
    def with_coupling(self,T):
        """Return a copy of this probe whose transparency is `T`, the
        same knob as LocalProbe(...,T=...) and set_coupling. Neither
        selfenergy depends on it (see get_selfenergy), so the copy keeps
        sharing the cache."""
        from copy import copy
        out = copy(self) # shallow, the Hamiltonians are shared
        out.T = T # transparency of the probe-sample coupling
        return out
    def get_reflection_normal_lead(self,s):
        return get_reflection_normal_lead(self,s)
    def didv(self,T=None,**kwargs):
        """Differential conductance. Routed through generic_didv (as
        Heterostructure.didv already is) rather than calling the bare
        method-selecting didv() directly, so that a `temp` kwarg here
        actually reaches transporttk.thermaldidv.finite_T_didv instead of
        being silently forwarded into a method (smatrix/keldysh) that
        never looks at it.

        `T` here is the probe TRANSPARENCY, the same knob as
        LocalProbe(...,T=...), set_coupling and get_kappa(T=...) -- NOT
        the temperature, which on Heterostructure.didv is what T spells.
        This class has both, and the transparency is the older meaning
        (examples/transport/didv_kitaev sweeps it through
        Hamiltonian.didv, which hands the same T to the constructor and
        to this method); the temperature here is `temp`, or its
        `temperature` alias. T used to be declared and never used, so
        neither reading happened and the probe's own transparency was
        silently used instead.

        At temp=0 (the default) this now goes through zero_T_didv, which
        defaults an unspecified delta to self.delta -- matching
        Heterostructure.didv's own convention -- rather than the bare
        didv()'s hardcoded delta=1e-6 that LocalProbe.didv used to fall
        through to before this routing existed. __init__'s own delta
        default is 1e-6 precisely so a caller who never touches delta
        anywhere still gets that same number; only a caller who
        constructs LocalProbe with an explicit delta=... now sees it
        consistently applied to didv() as well, rather than silently
        ignored in favor of 1e-6. Pass delta=... to didv() itself to
        override either default directly."""
        from .didv import generic_didv
        if T is not None and T!=self.T: # explicit, different transparency
            return generic_didv(self.with_coupling(T),**kwargs)
        return generic_didv(self,**kwargs)
    def didv_curve(self,energies,T=None,**kwargs):
        """Array-of-energies counterpart to `didv` above -- see
        transporttk.didv.didv_curve for the shared-AAA-interpolant
        behavior when `use_aaa=True` is passed. `T` is the probe
        transparency here too, for the same reasons as in didv."""
        from .didv import didv_curve
        if T is not None and T!=self.T: # explicit, different transparency
            return didv_curve(self.with_coupling(T),energies,**kwargs)
        return didv_curve(self,energies,**kwargs)
    def get_dc_current(self,voltage,**kwargs):
        """Floquet-Keldysh DC current at bias `voltage` between the probe
        and the sample site it couples to, see
        Heterostructure.get_dc_current and keldyshtk/current.py. Only
        meaningful (and only exact) when the probe lead is itself
        superconducting -- see didv(method="keldysh")."""
        from ..keldyshtk.current import dc_current
        return dc_current(self,voltage,**kwargs)
    def get_iv_curve(self,voltages,**kwargs):
        """Floquet-Keldysh I(V) curve, see get_dc_current"""
        from ..keldyshtk.current import iv_curve
        return iv_curve(self,voltages,**kwargs)
    def copy(self): return deepcopy(self)
    def set_coupling(self,c):
        self.T = c # set the coupling
    def remove_pairing(self):
        self.H.remove_pairing()
        self.lead.remove_pairing()
    def get_kappa(self,T=None,temp=0.,**kwargs):
        """Kappa (SC/normal conductance power-law ratio). temp=0 (default)
        keeps the original zero-temperature get_kappa_ratio behavior
        unchanged; temp!=0 routes through the thermally-averaged
        get_kappa_finite_temperature_energies instead (see
        Heterostructure.get_kappa's docstring for the same dispatch)."""
        if T is None: T = self.T
        if not temp:
            from .kappa import get_kappa_ratio
            return get_kappa_ratio(self,T=T,**kwargs)
        from .kappa import get_kappa_finite_temperature_energies
        single = "energies" not in kwargs
        energy = kwargs.pop("energy",0.0) # always pop: an explicit energy
        if single:                        # alongside energies must not be
            kwargs["energies"] = [energy] # forwarded twice further down
        out = get_kappa_finite_temperature_energies(self,T=T,temp=temp,**kwargs)
        return out[0] if single else out
    def get_dos(self,**kwargs):
        return get_dos_bulk(self,**kwargs)




def generate_gf(self,energy=0.0,numba=None,**kwargs):
    """Generate the specific Green's function"""
    mode = self.mode
    # just a trick to reuse the GF if needed
    if self.reuse_gf and self.gf is not None: return self.gf
    else:
        # forward `numba` down to bloch_selfenergy's Sancho-Rubio call (see
        # lead_selfenergy above for why: keldyshtk/current.py's
        # _cached_selfenergy passes numba=True to route the many thousands
        # of per-sideband calls in a Keldysh dc_current through the
        # compiled kernel instead of the slow pure-Python default)
        gf = self.H.get_gf(energy=energy,delta=self.bulk_delta,
                             mode=gfmode,
                             gtype=mode,
                             numba=numba)
        if self.reuse_gf: self.gf = gf # overwrite
        return gf


def generate_gf_batch(self,energies,**kwargs):
    """Batched twin of generate_gf: the sample's Green's function at every
    energy at once, as one (len(energies),n,n) array, through
    greentk.selfenergy.bloch_selfenergy_batch.

    A reused Green's function (`reuse_gf`) is honoured on the way in, the
    same energy-independent trick it is on the scalar path, but the batch
    never fills that slot: there is no single energy it would belong to."""
    if self.reuse_gf and self.gf is not None:
        gf = np.asarray(self.gf)
        return np.broadcast_to(gf,(len(energies),)+gf.shape)
    from ..greentk.selfenergy import bloch_selfenergy_batch
    return bloch_selfenergy_batch(self.H,energies,delta=self.bulk_delta,
                                   mode=gfmode,
                                   gtype=self.mode)[0]


def lead_selfenergy(self,energy=0.0,numba=None,**kwargs):
     """Return the selfenergy of the lead"""
     if self.frozen_lead: energy = 0.0 # set as zero energy
     delta = self.delta
     intra = self.lead.intra
     inter = dagger(self.lead.inter)
     cou = inter
     # forward `numba` to the Sancho-Rubio iteration: callers like
     # keldyshtk/current.py's _cached_selfenergy pass numba=True to route
     # through the compiled kernel (greentk.rg.green_renormalization_jit)
     # for a hot loop that recomputes this selfenergy many thousands of
     # times per dc_current call -- dropping it here (it used to land in
     # the swallowed **kwargs) silently forced the slow pure-Python path
     # every time regardless of what the caller asked for.
     ggg,g = green_renormalization(intra,inter,
                                     energy=energy,
                                     delta=delta,
                                     numba=numba)
     sigma = cou@g@dagger(cou) # selfenergy
     return sigma


def lead_selfenergy_batch(self,energies,**kwargs):
    """Batched twin of lead_selfenergy: the probe lead's selfenergy at
    every energy at once, as one (len(energies),dim,dim) array, through
    the numba prange-parallel decimation
    (greentk.rg.green_renormalization_jit_batch).

    A frozen lead is solved once and broadcast rather than resolved per
    entry -- freezing it means evaluating it at absolute zero energy
    whatever the bias (see lead_selfenergy), so every entry of the batch
    is the same matrix."""
    from ..greentk.rg import green_renormalization_jit_batch
    intra = self.lead.intra
    inter = dagger(self.lead.inter)
    cou = np.array(algebra.todense(inter)) # dense, for the batched matmul
    if self.frozen_lead: es = np.zeros(1) # all of them are this one
    else: es = np.asarray(energies,dtype=np.float64)
    ggg,g = green_renormalization_jit_batch(intra,inter,es,delta=self.delta)
    if self.frozen_lead:
        g = np.broadcast_to(g,(len(energies),)+g.shape[1:])
    return cou@g@dagger(cou) # selfenergy at every energy, batched matmul

from ..htk.extract import local_hamiltonian

def local_selfenergy(h,g,energy=0.0,i=0,delta=1e-5,**kwargs):
    """Given a certain Hamiltonian and Green's function, extract
    the local selfenergy"""
    M = get_intra(h) # get intracell matrix
    gi = local_hamiltonian(h,g,i=i) # local Green's function
    oi = local_hamiltonian(h,M,i=i) # local Hamiltonian
    iden = np.identity(gi.shape[0],dtype=np.complex128)
    out = algebra.inv(gi) - (energy+1j*delta)*iden + oi # local selfenergy
    return -out


def local_selfenergy_batch(h,g,energies,i=0,delta=1e-5,**kwargs):
    """Batched twin of local_selfenergy: `g` carries a leading energy axis
    and so does the result. The local block is a plain slice of each
    Green's function (htk.extract.local_hamiltonian, which this takes
    apart only so the slice is taken once for the whole batch rather than
    per energy), so the only real work left is one batched inversion."""
    from ..htk.extract import site_slice
    g = np.asarray(g)
    M = get_intra(h) # get intracell matrix
    s = site_slice(h,i)
    if s.stop>g.shape[-1]:
        raise ValueError("site "+str(i)+" lies outside a matrix of "
                "dimension "+str(g.shape[-1]))
    gi = g[:,s,s] # local Green's function at every energy
    oi = local_hamiltonian(h,M,i=i) # local Hamiltonian
    iden = np.identity(gi.shape[-1],dtype=np.complex128)
    e = (np.asarray(energies)+1j*delta)[:,None,None]*iden[None,:,:]
    out = np.linalg.inv(gi) - e + oi[None,:,:] # local selfenergy
    return -out



def get_central_gmatrix(P,selfl=None,selfr=None,energy=0.0):
    """Return the (inverse) central Green's function"""
    delta = P.delta # imaginary part
    if selfl is None: selfl = P.get_selfenergy(lead=0,energy=energy)
    if selfr is None: selfr = P.get_selfenergy(lead=1,energy=energy)
    if delta>delta_smatrix: delta = delta_smatrix # small delta is critical!
    iden = np.identity(selfl.shape[0],dtype=complex)*(energy +1j*delta)
    if P.frozen_lead:
        idenl = np.identity(selfl.shape[0],dtype=complex)*1j*delta
    else: idenl = iden
    hlist = [[None for i in range(2)] for j in range(2)] # list of matrices
    M = get_intra(P.H) # intracell matrix
    oi = local_hamiltonian(P.H,M,i=P.i) # local Hamiltonian
    # set up the different elements
    # first the intra terms
    hlist[0][0] = idenl - P.lead.intra - selfl
    hlist[1][1] = iden - oi - selfr
    # now the inter cell
    hlist[0][1] = -P.lead.inter*P.T # coupling times transparency
    hlist[1][0] = dagger(hlist[0][1]) # Hermitian conjugate
    return hlist



def get_dos_bulk(self,operator="electron",**kwargs):
    """Return the DOS of the bulk of a local probe object"""
    g = self.get_central_gmatrix(**kwargs) # return Green's function
    from ..green import gauss_inverse
    g11 = gauss_inverse(g,1,1)
    O = self.H.get_operator(operator)
    if O is not None:
        O = O*self.H.get_operator("site",index=0) # on a single site
        O = O.get_matrix() # get the matrix
        g11 = O@g11
    return -np.trace(g11.imag)/np.pi





def get_reflection_normal_lead(P,s):
    return s[0][0]




def get_intra(H):
    """Function to extract the intra-matrix from an object
    depending on its type"""
    from ..hamiltonians import Hamiltonian
    from ..embedding import Embedding
    if isinstance(H, Hamiltonian): return H.intra
    elif type(H)==Embedding: return H.m
    else:
        raise TypeError("the intracell matrix can only be extracted from a "
                "Hamiltonian or an Embedding")



def Hamiltonian_didv(self,**kwargs):
   """Wrapper to compute the didv for a Hamiltonian"""
   lp = LocalProbe(self,**kwargs) # create the object
   return lp.didv(**kwargs)


def Hamiltonian_didv_curve(self,energies,**kwargs):
   """Wrapper to compute didv_curve for a Hamiltonian"""
   lp = LocalProbe(self,**kwargs) # create the object
   return lp.didv_curve(energies,**kwargs)




