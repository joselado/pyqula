import numpy as np
from scipy.sparse import coo_matrix
from copy import deepcopy

# TODO
# - multi-species/q-state occupation (den is hard-restricted to exactly
#   2 distinct values in optimize_discrete); would need a rework of the
#   swap/flip moves and the energy formula, not just a parameter



class LatticeGas():
    def __init__(self,g,filling=0.5): # geometry
        g.nrep = 1
        self.geometry = g # store geometry
        self.nsites = len(g.r) # number of sites
        self.mu = np.zeros(len(g.r)) # chemical potential
        self.den = np.zeros(len(g.r)) # chemical potential
        self.j = np.array([0]) # interactions
        self.pairs = np.array([[0,0]]) # empty list
        self._adjacency = None # lazily-built cache, see _get_adjacency
        self.checkpoints = {} # populated by optimize_energy/anneal when
            # called with checkpoint_at, see their docstrings
        self.set_filling(filling)
    def set_filling(self,filling):
        """Set filling of the system"""
        self.den[:] = 0. # initialize all to zero
        N = int(np.round(self.nsites*filling)) # filled sites
        self.den = random_density(len(self.den),N) # random density
    def add_interaction(self,Jij=None,**kwargs):
        h = self.geometry.get_hamiltonian(has_spin=False,tij=Jij)
        m = coo_matrix(h.get_hk_gen()([0.,0.,0.])) # get onsite matrix
        pairs = np.array([m.row,m.col]).transpose() # convert to array
        self.pairs = np.concatenate([self.pairs,pairs]) # add interaction
        self.j = np.concatenate([self.j,m.data]).real # store
        self._adjacency = None # pairs/j changed, invalidate cache
    def _get_adjacency(self):
        """CSR adjacency of self.pairs/self.j, cached since it only
        depends on the (rarely-changing) interaction terms, not on the
        occupation snapshot den"""
        if self._adjacency is None:
            self._adjacency = _build_adjacency(self.nsites,self.pairs,self.j)
        return self._adjacency
    def get_energy(self):
        return energy_numba(self.mu,self.pairs,self.j,self.den)
    def get_local_energy(self,**kwargs):
        return get_local_energy(self,**kwargs)
    def get_local_mu(self,**kwargs):
        return get_local_mu(self,**kwargs)
    def optimize_energy(self,checkpoint_at=None,**kwargs):
        """Optimize the energy. `checkpoint_at`, if given (an int or
        iterable of ints), captures a copy of the configuration after
        that many trial moves (1-indexed: checkpoint_at=n is the state
        right after the n-th trial) into self.checkpoints (a dict
        step->den snapshot), independent of the final/returned state.
        temp=0 is supported (zero-temperature/greedy dynamics: an
        uphill move is never accepted)"""
        x,es,checkpoints = optimize_discrete(self.mu,self.pairs,self.j,self.den,
                adjacency=self._get_adjacency(),checkpoint_at=checkpoint_at,**kwargs)
        self.den = x # overwrite
        self.checkpoints = checkpoints
        return es
    def anneal(self,temps=None,ntries=1e4,checkpoint_at=None,**kwargs):
        """Simulated annealing over a decreasing temperature schedule,
        calling optimize_energy once per temperature. Keeps the best
        configuration seen across the whole schedule (a single high-T
        step's Metropolis walk can wander back up in energy by its
        end, so the last step's final state is not necessarily the
        best one found).

        `checkpoint_at`, if given (an int or iterable of ints),
        captures a copy of the configuration at those global trial
        indices (1-indexed, counting continuously across the whole
        schedule: temperature stage 1 covers steps 1..len(es_1),
        stage 2 continues from there, etc.) into self.checkpoints (a
        dict step->den snapshot), letting a caller recover the
        configuration after any number of annealing steps rather than
        only the final/best one"""
        if temps is None: temps = np.geomspace(2.0,0.05,10) # default cooling schedule
        steps = _normalize_checkpoint_steps(checkpoint_at)
        best_den = self.den.copy()
        best_e = self.get_energy()
        es = []
        checkpoints = {}
        offset = 0 # global trial count completed before the current stage
        for t in temps:
            local_steps = {s-offset for s in steps if offset<s<=offset+ntries}
            esi = self.optimize_energy(temp=t,ntries=ntries,
                    checkpoint_at=local_steps or None,**kwargs)
            for local_s,den_snap in self.checkpoints.items():
                checkpoints[local_s+offset] = den_snap
            es.append(esi)
            if esi[-1]<best_e:
                best_e = esi[-1]
                best_den = self.den.copy()
            offset += len(esi) # accounts for patience-truncated stages too
        self.den = best_den # restore the best configuration found
        self.checkpoints = checkpoints
        return np.concatenate(es)
    def optimize_energy_multistart(self,nstart=10,**kwargs):
        """Run nstart independent anneals from independent random
        seeds (same filling as the current lg.den) and keep the
        lowest-energy result, mirroring
        classicalspin.SpinModel.minimize_energy(tries=...). Each
        anneal is already a full numba-jitted loop (not itself
        further jittable), so restarts are farmed out with
        parallel.pcall rather than numba parallel=True; whether that
        is actually parallel depends on parallel.set_cores(), same as
        every other pcall call site in this package"""
        from . import parallel
        n = self.nsites
        mu,pairs,j = self.mu,self.pairs,self.j
        adjacency = self._get_adjacency()
        N = int(np.round(np.sum(self.den))) # keep the current filling
        seeds = np.random.randint(0,2**31-1,size=int(nstart))
        def run(seed):
            np.random.seed(seed) # decorrelate restarts sharing a worker process
            x0 = random_density(n,N)
            x,es,_ = optimize_discrete(mu,pairs,j,x0,adjacency=adjacency,**kwargs)
            return x,es[-1]
        results = parallel.pcall(run,seeds)
        x_best,e_best = min(results,key=lambda t: t[1])
        self.den = x_best
        return e_best
    def optimize_grand_canonical(self,**kwargs):
        """Grand-canonical Metropolis sampling/annealing: filling
        fluctuates under self.mu instead of being conserved (see the
        optimize_grand_canonical function docstring)"""
        x,es,ns = optimize_grand_canonical(self.mu,self.pairs,self.j,self.den,
                adjacency=self._get_adjacency(),**kwargs)
        self.den = x # overwrite
        return es,ns
    def add_tensor(self,fun):
        """Add a custom coupling J_ij = fun(r_i,r_j), for interactions
        beyond add_interaction()'s fixed neighbor shells (e.g.
        screened/dipolar long-range repulsion)"""
        pairs,js = add_tensor(self,fun)
        if len(pairs)==0: return # nothing matched
        self.pairs = np.concatenate([self.pairs,pairs])
        self.j = np.concatenate([self.j,js])
        self._adjacency = None # pairs/j changed, invalidate cache
    def regroup(self):
        """Merge duplicate pair entries accumulated from repeated
        add_interaction/add_tensor calls (see the regroup() function
        docstring). Pure performance cleanup: does not change
        get_energy()"""
        self.pairs,self.j = regroup(self.pairs,self.j)
        self._adjacency = None # pairs/j changed, invalidate cache
    def get_correlator(self,**kwargs):
        """Return the nearest neighbor correlators"""
        from .statphystk.correlator import get_nnc
        return get_nnc(self.geometry,self.den,**kwargs)
    def get_structure_factor(self,**kwargs):
        """Reciprocal-space structure factor S(q) of the current
        occupation snapshot -- the companion to get_correlator(): G(r)
        tells you the ordering length scale, S(q) tells you the
        ordering wavevector. See statphystk.correlator.get_structure_factor"""
        from .statphystk.correlator import get_structure_factor
        return get_structure_factor(self.geometry,self.den,**kwargs)
    def write(self,name="DENSITY.OUT",**kwargs):
        """Write the current occupation snapshot to a file, reusing
        Geometry.write_profile (see also
        classicalspin.SpinModel.write/load_magnetism, which follow
        the same checkpoint pattern for magnetization). nrep defaults
        to 1 (no periodic replication) so read() round-trips cleanly
        regardless of self.geometry.dimensionality"""
        kwargs.setdefault("nrep",1)
        self.geometry.write_profile(self.den,name=name,**kwargs)
    def read(self,name="DENSITY.OUT"):
        """Read a snapshot previously saved with write(), overwriting
        self.den. Rounds to {0,1} to absorb the text round-trip's
        floating point noise"""
        m = np.genfromtxt(name).transpose()
        den = np.round(m[2]) # third column is the profile (see write_profile)
        if len(den)!=self.nsites:
            raise ValueError(f"file {name} has {len(den)} sites, "
                f"expected {self.nsites}")
        self.den = den
    def copy(self):
        return deepcopy(self)

from numba import jit

def energy_numba(mu,pairs,js,den):
    if len(pairs)==0:
        pairs = np.array([[0,0]])
        js = np.array([0.])
    return energy_numba_jit(mu,pairs,js,den)


@jit(nopython=True)
def energy_numba_jit(mu,pairs,js,den):
    """Compute the energy of the lattice gas model"""
    nump = len(pairs) # number of pairs
    n = len(mu) # number of sites
    etot = 0. # output energy
    for i in range(n): etot = etot + mu[i]*den[i] # chemical potential
    for ip in range(nump):
        ii = pairs[ip][0]
        jj = pairs[ip][1]
        etot = etot + den[ii]*den[jj]*js[ip] # add contribution
    return etot



def _build_adjacency(n,pairs,js):
    """Build a CSR-style adjacency (rows keyed by pairs[:,0]) so that the
    interaction terms touching a given site can be found in O(degree)
    instead of scanning the full O(n_pairs) pairs array. Relies on the
    same convention as the rest of the module: pairs comes from a
    symmetric coupling matrix, so it always lists both (i,k) and (k,i)
    for every edge, and a site's own row already has full connectivity"""
    order = np.argsort(pairs[:,0],kind="stable")
    idx = pairs[:,1][order].astype(np.int64) # neighbor of each row entry
    jarr = js[order].astype(np.float64)
    ptr = np.searchsorted(pairs[:,0][order],np.arange(n+1)).astype(np.int64)
    return ptr,idx,jarr


@jit(nopython=True)
def _row_sum_excluding(ptr,idx,jarr,den,i,exclude):
    """sum_k J_{i,k}*den[k] over site i's neighbors, skipping `exclude`"""
    s = 0.
    for k in range(ptr[i],ptr[i+1]):
        neighbor = idx[k]
        if neighbor!=exclude:
            s += jarr[k]*den[neighbor]
    return s


@jit(nopython=True)
def swap_delta_energy(mu,ptr,idx,jarr,den,i1,i2):
    """Energy change from swapping den[i1] and den[i2]. Equivalent to
    energy_numba_jit(mu,pairs,js,den_after) - energy_numba_jit(mu,pairs,js,den_before)
    but costs O(degree(i1)+degree(i2)) instead of O(n_pairs): the direct
    i1-i2 edge (if any) cancels exactly under a swap since its
    contribution only depends on the product den[i1]*den[i2], which a
    swap leaves unchanged, so it is excluded from both row sums below"""
    a = den[i1] ; b = den[i2]
    if a==b: return 0. # nothing changes
    sum1 = _row_sum_excluding(ptr,idx,jarr,den,i1,i2)
    sum2 = _row_sum_excluding(ptr,idx,jarr,den,i2,i1)
    return (mu[i1]-mu[i2])*(b-a) + 2.*(b-a)*(sum1-sum2)


@jit(nopython=True)
def flip_delta_energy(mu,ptr,idx,jarr,den,i):
    """Energy change from flipping den[i] (0<->1) in place, the
    grand-canonical move complementing swap_delta_energy (filling is
    not conserved). `energy_numba_jit` double-counts every bond, once
    from each endpoint's own row (see the module-level docstring note
    in _build_adjacency and get_local_energy's docstring), so both the
    mu term and the row-sum term below carry a factor consistent with
    that convention: row already sums J_{i,k}*den[k] once per
    neighbor (single direction, via ptr[i]), and by the symmetric
    pairs guarantee that equals the "other direction" contribution
    too, hence the factor 2"""
    delta_n = 1.-2.*den[i] # +1 if turning on, -1 if turning off
    row = 0.
    for k in range(ptr[i],ptr[i+1]):
        row += jarr[k]*den[idx[k]]
    return mu[i]*delta_n + 2.*row*delta_n


def _normalize_checkpoint_steps(checkpoint_at):
    """Normalize checkpoint_at (None, a single int, or an iterable of
    ints) to a set of ints, used by optimize_discrete/anneal to decide
    at which trial indices to snapshot the configuration"""
    if checkpoint_at is None: return set()
    if np.isscalar(checkpoint_at): return {int(checkpoint_at)}
    return {int(s) for s in checkpoint_at}


def optimize_discrete(mu,pairs,js,x0,temp=0.1,ntries=1e5,info=False,
        resync_every=1000,adjacency=None,patience=None,checkpoint_at=None):
    """Discrete optimization, using a swap method. Energy changes are
    tracked incrementally with swap_delta_energy (O(degree) per swap)
    instead of recomputing the full O(n_pairs) energy on every try;
    the running energy is periodically resynced with a full recompute
    (every `resync_every` iterations) to bound floating-point drift.
    `adjacency`, the (ptr,idx,jarr) CSR tuple from _build_adjacency,
    can be passed in precomputed (e.g. cached across repeated calls
    with the same pairs/js, as LatticeGas.optimize_energy does) to
    skip rebuilding it; it only depends on pairs/js, not on x0.
    `patience`, if set, stops the loop early once `patience` tries
    have passed without a new best energy being found (`es` is
    truncated to what actually ran) -- the module's long-standing
    "autoannealing, stop once a reasonable GS is reached" TODO.
    temp=0 runs zero-temperature (greedy) dynamics: an uphill move
    (en>e) is never accepted, no Metropolis draw is made.
    `checkpoint_at`, if given (an int or iterable of ints), returns a
    copy of the configuration after that many trials (1-indexed) in
    the third return value, a dict step->den snapshot"""
    n = len(x0) # number of sites
    inds = np.arange(n) # indexes
    vals = np.unique(x0) # different values
    if len(vals)!=2:
        raise ValueError(f"optimize_discrete needs exactly 2 distinct "
            f"values in the density (got {vals}); check the filling "
            f"is not 0 or 1") # not implemented
    ptr,idx,jarr = adjacency if adjacency is not None else _build_adjacency(n,pairs,js)
    def pick_swap_pair(x):
        ainds = inds[x==vals[0]] # indexes for first value
        binds = inds[x==vals[1]] # indexes for second value
        j1 = np.random.randint(0,len(ainds)) # one random site
        j2 = np.random.randint(0,len(binds)) # one random site
        return ainds[j1],binds[j2]
    checkpoint_steps = _normalize_checkpoint_steps(checkpoint_at)
    checkpoints = {} # step (1-indexed) -> den snapshot
    nrep = int(ntries) # this many iterations
    x = x0.copy()
    e = energy_numba(mu,pairs,js,x) # current energy, tracked incrementally
    es = np.zeros(nrep) # storage for energies
    e_best = e ; i_best = 0 # for patience-based early stop
    for ii in range(nrep): # this many iterations
        if resync_every and ii%resync_every==0: # bound fp drift
            e = energy_numba(mu,pairs,js,x)
        xcand = x.copy()
        delta = 0.
        for _ in range(np.random.randint(1,4)): # make 1,2,3 swaps
            i1,i2 = pick_swap_pair(xcand)
            delta += swap_delta_energy(mu,ptr,idx,jarr,xcand,i1,i2)
            xcand[i1],xcand[i2] = xcand[i2],xcand[i1] # swap
        en = e + delta # new
        if info: print(en)
        accept = en<=e # smaller or same is always accepted
        if not accept and temp>0: # uphill move: Metropolis draw (never at temp=0)
            fac = np.exp((e-en)/temp) # acceptance probability
            accept = np.random.random()<fac
        if accept:
            es[ii] = en # store new energy
            x = xcand ; e = en # overwrite
        else:
            es[ii] = e # keep old energy
        if e<e_best: e_best = e ; i_best = ii
        if checkpoint_steps and (ii+1) in checkpoint_steps:
            checkpoints[ii+1] = x.copy()
        if patience is not None and ii-i_best>patience:
            es = es[0:ii+1] # only what actually ran
            break
    return x,es,checkpoints # return final state



def optimize_grand_canonical(mu,pairs,js,x0,temp=1.0,ntries=1e5,info=False,
        resync_every=1000,adjacency=None):
    """Grand-canonical Metropolis: single-site occupation flips
    (0<->1), accepted/rejected the usual way, so unlike
    optimize_discrete (fixed-filling swaps) the total filling is not
    conserved but fluctuates under the applied mu -- the standard
    lattice-gas MC move set, useful for scanning phase diagrams vs.
    chemical potential, or for equilibrium sampling at fixed temp (see
    get_specific_heat/get_susceptibility, which consume the es/ns
    trajectories returned here). Unlike optimize_discrete this has no
    2-distinct-values restriction: x0 can start uniform (all empty or
    all full). temp=0 runs zero-temperature (greedy) dynamics: a flip
    that raises the energy is never accepted"""
    n = len(x0) # number of sites
    ptr,idx,jarr = adjacency if adjacency is not None else _build_adjacency(n,pairs,js)
    nrep = int(ntries) # this many iterations
    x = x0.copy()
    e = energy_numba(mu,pairs,js,x) # current energy, tracked incrementally
    n_occ = np.sum(x) # current filling, tracked incrementally
    es = np.zeros(nrep) # storage for energies
    ns = np.zeros(nrep) # storage for filling
    for ii in range(nrep): # this many iterations
        if resync_every and ii%resync_every==0: # bound fp drift
            e = energy_numba(mu,pairs,js,x)
            n_occ = np.sum(x)
        i = np.random.randint(0,n) # random site to flip
        delta = flip_delta_energy(mu,ptr,idx,jarr,x,i)
        en = e + delta # new
        if info: print(en)
        accept = en<=e
        if not accept and temp>0: # uphill move: Metropolis draw (never at temp=0)
            fac = np.exp((e-en)/temp) # acceptance probability
            accept = np.random.random()<fac
        if accept:
            n_occ += 1.-2.*x[i] # +1 if turning on, -1 if turning off
            x[i] = 1.-x[i] # flip
            e = en
        es[ii] = e
        ns[ii] = n_occ
    return x,es,ns # return final state



def get_specific_heat(es,temp,burn=0.2):
    """Estimate the specific heat C=Var(E)/T^2 from an energy
    trajectory `es` sampled at fixed temperature `temp` (e.g. from
    optimize_energy or optimize_grand_canonical run at one fixed
    temp, not annealed), discarding the first `burn` fraction of
    steps as equilibration"""
    n0 = int(len(es)*burn)
    return np.var(es[n0:])/temp**2


def get_susceptibility(ns,temp,burn=0.2):
    """Estimate the particle-number susceptibility (compressibility)
    dN/dmu=Var(N)/T from a filling trajectory `ns` sampled at fixed
    temperature `temp` in the grand-canonical ensemble (the `ns`
    returned by optimize_grand_canonical), discarding the first `burn`
    fraction of steps as equilibration"""
    n0 = int(len(ns)*burn)
    return np.var(ns[n0:])/temp



@jit(nopython=True)
def _row_sums_den_numba(ptr,idx,jarr,den):
    """sum_k J_ik*den[k] over each site's neighbors, via the CSR
    adjacency -- the row sum shared by get_local_energy/get_local_mu"""
    n = len(ptr)-1
    out = np.zeros(n)
    for i in range(n):
        row = 0.
        for k in range(ptr[i],ptr[i+1]):
            row += jarr[k]*den[idx[k]]
        out[i] = row
    return out


@jit(nopython=True)
def _row_sums_numba(ptr,jarr):
    """sum_k J_ik over each site's neighbors, the normalize=True
    divisor for get_local_energy/get_local_mu"""
    n = len(ptr)-1
    out = np.zeros(n)
    for i in range(n):
        row = 0.
        for k in range(ptr[i],ptr[i+1]):
            row += jarr[k]
        out[i] = row
    return out


def get_local_energy(LG,normalize=False):
    """Return the local energy at each site for the current snapshot.

    Previously built, per site, a boolean mask over the *entire*
    pairs array and called the whole-lattice energy_numba (itself an
    O(n) loop over all sites for the mu term) -- O(n*n_pairs) total.
    Now walks the cached CSR adjacency once, O(n_pairs) total -- see
    latticeising.get_local_energy, which mirrors this same fix"""
    ptr,idx,jarr = LG._get_adjacency()
    row = _row_sums_den_numba(ptr,idx,jarr,LG.den)
    out = LG.mu*LG.den + LG.den*row
    if normalize: out = out/_row_sums_numba(ptr,jarr)
    return out


def get_local_mu(LG,normalize=False):
    """Return the local chemical potential: the marginal energy cost
    of occupying site ii, independent of its current occupation (mu_i
    + sum_k J_ik*den_k) -- see get_local_energy's docstring for the
    same O(n*n_pairs) -> O(n_pairs) rewrite"""
    ptr,idx,jarr = LG._get_adjacency()
    row = _row_sums_den_numba(ptr,idx,jarr,LG.den)
    out = LG.mu + row
    if normalize: out = out/_row_sums_numba(ptr,jarr)
    return out



def random_density(Ntot,N):
    """Generate an array with N 1's and Ntot-N 0's,
    with the 1's randomly distributed"""
    out = np.zeros(Ntot) # initialize
    inds = np.random.choice(Ntot,N,replace=False) # N random indexes
    out[inds] = 1.0
    return out



def add_tensor(LG,fun):
    """Add a custom coupling J_ij=fun(r_i,r_j) between every pair of
    sites, for interactions beyond add_interaction()'s fixed neighbor
    shells (e.g. screened/dipolar long-range repulsion). Scalar analog
    of classicalspin.add_tensor (which returns a 3x3 tensor per pair);
    self-pairs (i==i) are skipped since fun is typically a
    distance-based form (e.g. 1/r) that is not defined at r=0, and the
    model has no onsite n_i^2 term anyway (mu already covers the
    onsite/chemical-potential piece)"""
    pairs = []
    js = []
    r = LG.geometry.r
    n = LG.nsites
    for i1 in range(n):
        for i2 in range(n):
            if i1==i2: continue
            jij = fun(r[i1],r[i2])
            if abs(jij)>1e-7: # if non zero
                pairs.append((i1,i2))
                js.append(jij)
    return np.array(pairs),np.array(js)



def regroup(pairs,js):
    """Merge exactly-duplicate pair entries (e.g. from calling
    add_interaction/add_tensor more than once with overlapping
    shells), summing their couplings. Unlike
    classicalspin.regroup, (i,j) and (j,i) are kept as separate
    entries rather than folded into one: _build_adjacency's per-site
    row sums rely on both directions being present, each in its own
    row (see its docstring), so folding them would silently drop
    connectivity from one endpoint's row"""
    dictj = dict() # accumulate contributions, keyed by ordered pair
    for (p,j) in zip(pairs,js): # loop over inputs
        key = (p[0],p[1])
        dictj[key] = dictj.get(key,0.)+j # add contribution
    outp = list(dictj.keys()) # dicts preserve insertion order
    outj = [dictj[p] for p in outp] # get all
    return np.array(outp),np.array(outj)



