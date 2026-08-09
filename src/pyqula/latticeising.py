import numpy as np
from scipy.sparse import coo_matrix
from copy import deepcopy
from numba import jit

from .latticegas import (_build_adjacency, _row_sum_excluding,
        _normalize_checkpoint_steps, add_tensor as _add_tensor_gas,
        regroup as _regroup_pairs)

# Mirrors latticegas.LatticeGas's structure (geometry-driven pair list,
# CSR adjacency cache, Metropolis swap/flip optimizers, checkpointing),
# adapted from occupation numbers den in {0,1} to Ising spins s in
# {-1,+1}. The two models use OPPOSITE energy sign conventions: LatticeGas
# uses E = sum mu*den + sum J*den_i*den_j (no minus sign, so positive J is
# a *repulsion*), while LatticeIsing follows the standard textbook Ising
# Hamiltonian E = -sum b*s - sum J*s_i*s_j (positive J is *ferromagnetic*,
# favoring alignment). Since self.pairs lists both (i,j) and (j,i) for
# every bond (see _build_adjacency), get_energy() is twice the usual
# sum-over-unordered-bonds convention -- e.g. the 2d square-lattice
# ferromagnet's critical temperature lands at 2*2.269, not 2.269, in
# these units. Don't try to divide it away: get_local_energy() below
# relies on the same j/2 correction latticegas uses, and halving get_energy
# would break that mirror.


class LatticeIsing():
    def __init__(self,g,m=0.0): # geometry, initial magnetization
        g.nrep = 1
        self.geometry = g # store geometry
        self.nsites = len(g.r) # number of sites
        self.b = np.zeros(len(g.r)) # external field
        self.s = np.ones(len(g.r)) # spin configuration, +-1
        self.j = np.array([0.]) # interactions
        self.pairs = np.array([[0,0]]) # empty list
        self._adjacency = None # lazily-built cache, see _get_adjacency
        self.checkpoints = {} # populated by optimize_energy/optimize_conserved/anneal
            # when called with checkpoint_at, see their docstrings
        self.set_magnetization(m)
    def set_magnetization(self,m=0.0):
        """Reset self.s to a new random +-1 configuration with average
        magnetization approximately equal to m (in [-1,1])"""
        N_up = int(np.round(self.nsites*(1.+m)/2.))
        self.s = random_spins(self.nsites,N_up)
    def add_interaction(self,Jij=None,**kwargs):
        h = self.geometry.get_hamiltonian(has_spin=False,tij=Jij)
        m = coo_matrix(h.get_hk_gen()([0.,0.,0.])) # get onsite matrix
        pairs = np.array([m.row,m.col]).transpose() # convert to array
        self.pairs = np.concatenate([self.pairs,pairs]) # add interaction
        self.j = np.concatenate([self.j,m.data]).real # store
        self._adjacency = None # pairs/j changed, invalidate cache
    def add_field(self,h):
        """Add an external (Zeeman-like) field: either a scalar
        (applied uniformly) or a per-site array"""
        self.b = self.b + np.zeros(self.nsites) + h
    def _get_adjacency(self):
        """CSR adjacency of self.pairs/self.j, cached since it only
        depends on the (rarely-changing) interaction terms, not on the
        spin snapshot self.s -- see latticegas.LatticeGas._get_adjacency"""
        if self._adjacency is None:
            self._adjacency = _build_adjacency(self.nsites,self.pairs,self.j)
        return self._adjacency
    def get_energy(self):
        return ising_energy_numba(self.b,self.pairs,self.j,self.s)
    def get_magnetization(self):
        return np.mean(self.s)
    def get_local_energy(self,**kwargs):
        return get_local_energy(self,**kwargs)
    def get_local_field(self,**kwargs):
        return get_local_field(self,**kwargs)
    def optimize_energy(self,checkpoint_at=None,**kwargs):
        """Single-spin-flip Metropolis dynamics (the standard Ising MC
        move set): magnetization is NOT conserved, fluctuating instead
        under self.b -- the spin analog of
        latticegas.LatticeGas.optimize_grand_canonical, which is why
        (like that method, and unlike optimize_conserved below) this
        returns (es, ms): the energy trajectory and the total
        magnetization (sum of self.s) trajectory. temp=0 supported
        (zero-temperature/greedy dynamics). No `patience`: the
        trajectory this returns is meant to be fed to
        latticegas.get_specific_heat/get_susceptibility, and early
        truncation would silently bias those variance estimates"""
        x,es,ms,checkpoints = optimize_ising(self.b,self.pairs,self.j,self.s,
                adjacency=self._get_adjacency(),checkpoint_at=checkpoint_at,**kwargs)
        self.s = x # overwrite
        self.checkpoints = checkpoints
        return es,ms
    def optimize_conserved(self,checkpoint_at=None,**kwargs):
        """Kawasaki spin-exchange dynamics: swap two opposite-sign
        spins, which conserves the total magnetization -- the
        canonical-ensemble analog of optimize_energy's single-flip
        moves, mirroring latticegas.LatticeGas.optimize_energy
        (swap-based, fixed filling) vs. optimize_grand_canonical
        (flip-based, fluctuating filling). Requires self.s to have
        both +1 and -1 present (raises ValueError otherwise, e.g. if
        set_magnetization(1.0) left every spin up)"""
        x,es,checkpoints = optimize_ising_conserved(self.b,self.pairs,self.j,self.s,
                adjacency=self._get_adjacency(),checkpoint_at=checkpoint_at,**kwargs)
        self.s = x # overwrite
        self.checkpoints = checkpoints
        return es
    def anneal(self,temps=None,ntries=1e4,checkpoint_at=None,**kwargs):
        """Simulated annealing over a decreasing temperature schedule,
        calling optimize_energy once per temperature. Keeps the best
        (lowest-energy) configuration seen across the whole schedule,
        same rationale as latticegas.LatticeGas.anneal.

        `checkpoint_at`, if given (an int or iterable of ints),
        captures a copy of the configuration at those global trial
        indices (1-indexed, counting continuously across the whole
        schedule) into self.checkpoints (a dict step->s snapshot)"""
        if temps is None: temps = np.geomspace(2.0,0.05,10) # default cooling schedule
        steps = _normalize_checkpoint_steps(checkpoint_at)
        best_s = self.s.copy()
        best_e = self.get_energy()
        es = [] ; ms = []
        checkpoints = {}
        offset = 0 # global trial count completed before the current stage
        for t in temps:
            local_steps = {s-offset for s in steps if offset<s<=offset+ntries}
            esi,msi = self.optimize_energy(temp=t,ntries=ntries,
                    checkpoint_at=local_steps or None,**kwargs)
            for local_s,s_snap in self.checkpoints.items():
                checkpoints[local_s+offset] = s_snap
            es.append(esi) ; ms.append(msi)
            if esi[-1]<best_e:
                best_e = esi[-1]
                best_s = self.s.copy()
            offset += len(esi)
        self.s = best_s # restore the best configuration found
        self.checkpoints = checkpoints
        return np.concatenate(es),np.concatenate(ms)
    def optimize_energy_multistart(self,nstart=10,**kwargs):
        """Run nstart independent flip-based anneals from independent
        random seeds (same initial magnetization as the current
        self.s) and keep the lowest-energy result, mirroring
        latticegas.LatticeGas.optimize_energy_multistart"""
        from . import parallel
        n = self.nsites
        b,pairs,j = self.b,self.pairs,self.j
        adjacency = self._get_adjacency()
        N_up = int(np.round(np.sum(self.s==1.)))
        seeds = np.random.randint(0,2**31-1,size=int(nstart))
        def run(seed):
            np.random.seed(seed) # decorrelate restarts sharing a worker process
            x0 = random_spins(n,N_up)
            x,es,ms,_ = optimize_ising(b,pairs,j,x0,adjacency=adjacency,**kwargs)
            return x,es[-1]
        results = parallel.pcall(run,seeds)
        x_best,e_best = min(results,key=lambda t: t[1])
        self.s = x_best
        return e_best
    def add_tensor(self,fun):
        """Add a custom coupling J_ij = fun(r_i,r_j), for interactions
        beyond add_interaction()'s fixed neighbor shells (e.g.
        screened/dipolar long-range exchange). Reuses
        latticegas.add_tensor directly: that function only touches
        LG.geometry.r and LG.nsites, both present here with identical
        meaning"""
        pairs,js = _add_tensor_gas(self,fun)
        if len(pairs)==0: return # nothing matched
        self.pairs = np.concatenate([self.pairs,pairs])
        self.j = np.concatenate([self.j,js])
        self._adjacency = None # pairs/j changed, invalidate cache
    def regroup(self):
        """Merge duplicate pair entries accumulated from repeated
        add_interaction/add_tensor calls. Reuses latticegas.regroup
        directly (a pure function of pairs/j, not gas-specific)"""
        self.pairs,self.j = _regroup_pairs(self.pairs,self.j)
        self._adjacency = None # pairs/j changed, invalidate cache
    def get_correlator(self,**kwargs):
        """Spin-spin correlator, reusing statphystk.correlator.get_nnc
        (generic over any per-site array, not specific to occupation
        numbers)"""
        from .statphystk.correlator import get_nnc
        return get_nnc(self.geometry,self.s,**kwargs)
    def get_structure_factor(self,**kwargs):
        """Reciprocal-space structure factor S(q) of the current spin
        snapshot, reusing statphystk.correlator.get_structure_factor --
        see latticegas.LatticeGas.get_structure_factor"""
        from .statphystk.correlator import get_structure_factor
        return get_structure_factor(self.geometry,self.s,**kwargs)
    def write(self,name="SPIN.OUT",**kwargs):
        """Write the current spin snapshot to a file, reusing
        Geometry.write_profile -- see latticegas.LatticeGas.write"""
        kwargs.setdefault("nrep",1)
        self.geometry.write_profile(self.s,name=name,**kwargs)
    def read(self,name="SPIN.OUT"):
        """Read a snapshot previously saved with write(), overwriting
        self.s. Rounds to {-1,+1} (sign, treating exact 0 as +1) to
        absorb the text round-trip's floating point noise"""
        m = np.genfromtxt(name).transpose()
        s = np.where(m[2]>=0.,1.,-1.) # third column is the profile
        if len(s)!=self.nsites:
            raise ValueError(f"file {name} has {len(s)} sites, "
                f"expected {self.nsites}")
        self.s = s
    def copy(self):
        return deepcopy(self)


def ising_energy_numba(b,pairs,js,s):
    if len(pairs)==0:
        pairs = np.array([[0,0]])
        js = np.array([0.])
    return ising_energy_numba_jit(b,pairs,js,s)


@jit(nopython=True)
def ising_energy_numba_jit(b,pairs,js,s):
    """Compute the energy of the Ising model, E = -sum_i b_i*s_i -
    sum_pairs J_ij*s_i*s_j (pairs lists both directions of every bond,
    so this is twice the usual sum-over-unordered-bonds convention)"""
    nump = len(pairs) # number of pairs
    n = len(b) # number of sites
    etot = 0. # output energy
    for i in range(n): etot = etot - b[i]*s[i] # field term
    for ip in range(nump):
        ii = pairs[ip][0]
        jj = pairs[ip][1]
        etot = etot - s[ii]*s[jj]*js[ip] # add contribution
    return etot


@jit(nopython=True)
def flip_delta_energy_ising(b,ptr,idx,jarr,s,i):
    """Energy change from flipping s[i] -> -s[i] in place, the
    grand-canonical-like move complementing swap_delta_energy_ising
    (magnetization is not conserved). Equal to
    2*s[i]*get_local_field()[i] -- see that function's docstring for
    the derivation"""
    row = 0.
    for k in range(ptr[i],ptr[i+1]):
        row += jarr[k]*s[idx[k]]
    return 2.*b[i]*s[i] + 4.*s[i]*row


@jit(nopython=True)
def swap_delta_energy_ising(b,ptr,idx,jarr,s,i1,i2):
    """Energy change from swapping s[i1] and s[i2] (equivalently,
    flipping both simultaneously, since Ising spins only take 2
    values). The direct i1-i2 bond's contribution depends only on
    s[i1]*s[i2], which a simultaneous double-flip leaves invariant, so
    (as in latticegas.swap_delta_energy) it is excluded from both row
    sums below"""
    a = s[i1] ; b_ = s[i2]
    if a==b_: return 0. # nothing changes
    row1 = _row_sum_excluding(ptr,idx,jarr,s,i1,i2)
    row2 = _row_sum_excluding(ptr,idx,jarr,s,i2,i1)
    return 2.*b[i1]*s[i1] + 2.*b[i2]*s[i2] + 4.*s[i1]*row1 + 4.*s[i2]*row2


def optimize_ising(b,pairs,js,s0,temp=1.0,ntries=1e5,info=False,
        resync_every=1000,adjacency=None,checkpoint_at=None):
    """Single-spin-flip Metropolis dynamics: at each step, one random
    site is flipped and accepted unconditionally if the energy doesn't
    increase, or with probability e^{-dE/T} otherwise. temp=0 runs
    zero-temperature (greedy) dynamics. Mirrors
    latticegas.optimize_grand_canonical (no fixed-filling restriction,
    magnetization fluctuates), returning (s,es,ms,checkpoints) where
    ms is the total-magnetization (sum of s) trajectory, meant to feed
    latticegas.get_susceptibility"""
    n = len(s0) # number of sites
    ptr,idx,jarr = adjacency if adjacency is not None else _build_adjacency(n,pairs,js)
    checkpoint_steps = _normalize_checkpoint_steps(checkpoint_at)
    checkpoints = {} # step (1-indexed) -> s snapshot
    nrep = int(ntries) # this many iterations
    x = s0.copy()
    e = ising_energy_numba(b,pairs,js,x) # current energy, tracked incrementally
    m_tot = np.sum(x) # current magnetization, tracked incrementally
    es = np.zeros(nrep) # storage for energies
    ms = np.zeros(nrep) # storage for magnetization
    for ii in range(nrep): # this many iterations
        if resync_every and ii%resync_every==0: # bound fp drift
            e = ising_energy_numba(b,pairs,js,x)
            m_tot = np.sum(x)
        i = np.random.randint(0,n) # random site to flip
        delta = flip_delta_energy_ising(b,ptr,idx,jarr,x,i)
        en = e + delta # new
        if info: print(en)
        accept = en<=e # smaller or same is always accepted
        if not accept and temp>0: # uphill move: Metropolis draw (never at temp=0)
            fac = np.exp((e-en)/temp) # acceptance probability
            accept = np.random.random()<fac
        if accept:
            m_tot += -2.*x[i] # +1 -> -1 subtracts 2, -1 -> +1 adds 2
            x[i] = -x[i] # flip
            e = en
        es[ii] = e
        ms[ii] = m_tot
        if checkpoint_steps and (ii+1) in checkpoint_steps:
            checkpoints[ii+1] = x.copy()
    return x,es,ms,checkpoints # return final state



def optimize_ising_conserved(b,pairs,js,x0,temp=0.1,ntries=1e5,info=False,
        resync_every=1000,adjacency=None,patience=None,checkpoint_at=None):
    """Kawasaki spin-exchange dynamics: at each step, one +1 site and
    one -1 site are picked at random and swapped, conserving the total
    magnetization. Mirrors latticegas.optimize_discrete (swap-based,
    fixed order parameter), including its `patience` early-stop and
    `checkpoint_at` snapshotting"""
    n = len(x0) # number of sites
    inds = np.arange(n) # indexes
    vals = np.unique(x0) # different values
    if len(vals)!=2:
        raise ValueError(f"optimize_ising_conserved needs exactly 2 "
            f"distinct spin values present (got {vals}); check the "
            f"magnetization is not +-1 (fully polarized)")
    ptr,idx,jarr = adjacency if adjacency is not None else _build_adjacency(n,pairs,js)
    def pick_swap_pair(x):
        ainds = inds[x==vals[0]] # indexes for first value
        binds = inds[x==vals[1]] # indexes for second value
        j1 = np.random.randint(0,len(ainds)) # one random site
        j2 = np.random.randint(0,len(binds)) # one random site
        return ainds[j1],binds[j2]
    checkpoint_steps = _normalize_checkpoint_steps(checkpoint_at)
    checkpoints = {} # step (1-indexed) -> s snapshot
    nrep = int(ntries) # this many iterations
    x = x0.copy()
    e = ising_energy_numba(b,pairs,js,x) # current energy, tracked incrementally
    es = np.zeros(nrep) # storage for energies
    e_best = e ; i_best = 0 # for patience-based early stop
    for ii in range(nrep): # this many iterations
        if resync_every and ii%resync_every==0: # bound fp drift
            e = ising_energy_numba(b,pairs,js,x)
        i1,i2 = pick_swap_pair(x)
        delta = swap_delta_energy_ising(b,ptr,idx,jarr,x,i1,i2)
        en = e + delta # new
        if info: print(en)
        accept = en<=e # smaller or same is always accepted
        if not accept and temp>0: # uphill move: Metropolis draw (never at temp=0)
            fac = np.exp((e-en)/temp) # acceptance probability
            accept = np.random.random()<fac
        if accept:
            x[i1],x[i2] = x[i2],x[i1] # swap
            e = en
        es[ii] = e
        if e<e_best: e_best = e ; i_best = ii
        if checkpoint_steps and (ii+1) in checkpoint_steps:
            checkpoints[ii+1] = x.copy()
        if patience is not None and ii-i_best>patience:
            es = es[0:ii+1] # only what actually ran
            break
    return x,es,checkpoints # return final state



def random_spins(Ntot,N_up):
    """Generate an array with N_up +1's and Ntot-N_up -1's, with the
    +1's randomly distributed -- spin analog of latticegas.random_density"""
    out = -np.ones(Ntot) # initialize, all down
    inds = np.random.choice(Ntot,N_up,replace=False) # N_up random indexes
    out[inds] = 1.0
    return out



@jit(nopython=True)
def _local_energy_numba(b,ptr,idx,jarr,s):
    """Local energy at every site, en[i] = -b[i]*s[i] -
    s[i]*sum_k J_ik*s[k], via the CSR adjacency (O(degree) per site,
    O(n_pairs) total) instead of masking the full pairs array and
    re-running the whole-lattice energy sum per site -- see
    get_local_energy's docstring for why this replaced that approach"""
    n = len(b)
    out = np.zeros(n)
    for i in range(n):
        row = 0.
        for k in range(ptr[i],ptr[i+1]):
            row += jarr[k]*s[idx[k]]
        out[i] = -b[i]*s[i] - s[i]*row
    return out


@jit(nopython=True)
def _row_sums_numba(ptr,jarr):
    """sum_k J_ik over each site's neighbors, the normalize=True
    divisor for get_local_energy"""
    n = len(ptr)-1
    out = np.zeros(n)
    for i in range(n):
        row = 0.
        for k in range(ptr[i],ptr[i+1]):
            row += jarr[k]
        out[i] = row
    return out


def get_local_energy(LI,normalize=False):
    """Return the local energy at each site for the current snapshot.

    Previously built, per site, a boolean mask over the *entire*
    pairs array and called the whole-lattice ising_energy_numba (itself
    an O(n) loop over all sites for the field term) -- O(n*n_pairs)
    total. Now walks the cached CSR adjacency once, O(n_pairs) total,
    the same approach get_local_field already used"""
    ptr,idx,jarr = LI._get_adjacency()
    out = _local_energy_numba(LI.b,ptr,idx,jarr,LI.s)
    if normalize: out = out/_row_sums_numba(ptr,jarr)
    return out


@jit(nopython=True)
def _local_field_numba(b,ptr,idx,jarr,s):
    n = len(b)
    out = np.zeros(n)
    for i in range(n):
        row = 0.
        for k in range(ptr[i],ptr[i+1]):
            row += jarr[k]*s[idx[k]]
        out[i] = b[i] + 2.*row
    return out


def get_local_field(LI):
    """Return the effective field h_eff_i = b_i + 2*sum_k J_ik*s_k
    seen by each site, i.e. the field such that flipping s_i costs
    exactly 2*s_i*h_eff_i (the factor of 2 in front of the sum, unlike
    the factor of 1 you might naively expect, comes from pairs listing
    both (i,k) and (k,i) -- same double-counting convention as
    get_energy())"""
    ptr,idx,jarr = LI._get_adjacency()
    return _local_field_numba(LI.b,ptr,idx,jarr,LI.s)
