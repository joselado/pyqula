import numpy as np
from numba import jit



def get_mf_anomalous(v,dm):
    """Compute the anomalous part of the mean-field,
    input in the anomalous density matrix in Nambu basis"""
    zero = dm[(0,0,0)]*0. # zero
    mf = dict()
    for d in v: mf[d] = zero.copy()  # initialize
    for d in v: # loop over directions
        d2 = tuple(-np.array(d))
        m = anomalous_term_ij(2*v[d],dm[d2]) # get matrix
        mf[d] = mf[d] + m # add normal term
    return mf



def anomalous_term_ij(v,dm):
    """Return the anomalous term of the mean-field,
    assuming a nmabu basis"""
    # we will assume that v contains up,down in alternating order
    n = dm.shape[0] # number of spinless sites
    out = np.zeros((n,n),dtype=np.complex128)
    return anomalous_term_ij_jit(v,dm,out)



# this anomalous term enforces even superconductivity

@jit(nopython=True,cache=True)
def anomalous_term_ij_jit(v,dm,out):
    ns = len(dm)//2 # number of spinless sites
    for i in range(ns): # loop over spinless sites
        for j in range(ns): # loop over spinless sites
          out[2*i,2*j] = v[2*i,2*j+1]*dm[2*j,2*i]  # down,up
          out[2*i,2*j+1] = v[2*i,2*j]*dm[2*j+1,2*i]  # up,up
          out[2*i+1,2*j+1] = v[2*i+1,2*j]*dm[2*j+1,2*i+1]  # up,down
          out[2*i+1,2*j] = v[2*i+1,2*j+1]*dm[2*j,2*i+1]  # down,down
    return out



def get_dc_energy_anomalous(mf,dm):
    """Double-counting energy of the anomalous (pairing) mean field.

    mf: Nambu mean-field dict, as returned by get_mf_bdg (its electron and
    hole blocks are ignored, only the anomalous blocks are read).
    dm: Nambu density matrix dict, as returned by h.get_density_matrix
    (dm[i,j] = sum_occ conj(psi_i) psi_j, the transpose of the usual one).

    The anomalous part of the interaction energy is a bilinear form in the
    anomalous density matrix, and get_mf_bdg's anomalous mean field is its
    derivative, so its expectation value counts that energy exactly twice.
    The band energy of the BdG Hamiltonian therefore contains it twice and
    this returns minus it once: -<MF_anomalous>/2, with
    <X> = (1/2) sum_d Tr(dm[d].T X[d]) for a Nambu matrix, and no constant
    term since the anomalous blocks carry no diagonal. It is the pairing
    counterpart of densitydensity.get_dc_energy, which only accounts for
    the Hartree and Fock terms.

    Only the (0,1) block is read. Both mf and dm are Hermitian in real
    space (X[d] = X[-d]^dagger, and mf holds every d together with -d), so
    the (1,0) block contributes the complex conjugate of the (0,1) one,
    and the sparse density matrices of the KPM and qtci backends fill in
    only the (0,1) block, the one get_mf_bdg reads."""
    from ..superconductivity import get_eh_sector
    out = 0.0
    for d in mf: # loop over directions
        m = get_eh_sector(np.asarray(mf[d]),i=0,j=1)
        if not np.any(m): continue # no pairing in this direction
        out += np.sum(get_eh_sector(np.asarray(dm[d]),i=0,j=1)*m)
    return -out.real/2.



def enforce_eh_symmetry_anomalous(d01):
    """Enforce electron-hole symmetry in the two sectors"""
    d01 = enforce_eh_symmetry_anomalous_sector(d01)
    d10 = enforce_eh_from_sector(d01)
    return d01,d10



def enforce_eh_from_sector(d):
    """Given one sector of the Hamiltonian, return the other one"""
    out = dict() # dictionary
    for key in d:
        m = d[key] # one key
        o01 = m*0.0
        key2 = tuple(-np.array(key)) # the opposite
        out[key2] = enforce_eh_from_sector_jit(m,o01)
    return out # return


@jit(nopython=True,cache=True)
def enforce_eh_from_sector_jit(d,o):
    """Given the ee sector, return the hh sector"""
    return np.conjugate(d.T) # hermitian conjugate


def enforce_eh_symmetry_anomalous_sector(d01):
    """Enforce electron-hole symmetry in one of the sectors"""
    out01 = dict()
    for key in d01: # loop over keys
        key2 = tuple(-np.array(key)) # minus one
        o01 = d01[key]*0.0
        o01 = enforce_eh_symmetry_anomalous_jit(d01[key],d01[key2],o01)
        out01[key] = o01
    return out01



@jit(nopython=True,cache=True)
def enforce_eh_symmetry_anomalous_jit(d01,d10,o01):
    """Enforce electron-hole symmetry"""
    ns = len(d01)//2 # number of spinless sites
    for i in range(ns): # loop
        for j in range(ns): # loop
            # enforce the up|up sector
            o01[2*i,2*j+1] = d01[2*i,2*j+1] - d10[2*j,2*i+1]
            # enforce the down|down sector1
            o01[2*i+1,2*j] = d01[2*i+1,2*j] - d10[2*j+1,2*i]
            # enforce the up|down sector1 (beware of the minus sign)
            o01[2*i,2*j] = d01[2*i,2*j] + d10[2*j+1,2*i+1]
            o01[2*i+1,2*j+1] = d01[2*i+1,2*j+1] + d10[2*j,2*i]
    return o01/2.



def get_mf_bdg(v,dm,compute_anomalous=True,compute_normal=True,**kwargs):
    """Get the full BdG (Nambu) mean-field matrix.

    v: interaction matrix dict, spin-orbital sized (NOT Nambu-doubled) --
    the same convention densitydensity.get_mf_normal takes.
    dm: Nambu-doubled density matrix dict, as returned by diagonalizing a
    BdG Hamiltonian (h.has_eh=True).

    Same v/dm-in, mf-out contract as get_mf_normal(v,dm) -- this is the
    has_eh=True counterpart called by densitydensity.get_mf -- but
    internally it has to combine two genuinely different Wick-contraction
    topologies of the same density-density interaction: the normal
    (Hartree+Fock) decoupling of the electron sector <c^+c> (get_mf_normal,
    reused as-is) and the anomalous (pairing) decoupling of the <cc> sector
    (get_mf_anomalous, this module). These are not the same formula run on
    different data -- compare anomalous_term_ij_jit's index gymnastics
    against normal_term_ij_jit's -- so neither can be dropped in favor of
    the other; this function's job is only to package the two into one
    self-contained BdG decoupling step."""
    from .. import superconductivity
    from ..multihopping import MultiHopping
    from .densitydensity import get_mf_normal
    dme = dict() # electron sector of the density matrix, one per direction
    dma10 = dict() # anomalous (electron-hole) sector
    for key in dm:
        m = dm[key]
        dme[key] = superconductivity.get_eh_sector(m,i=0,j=0)
        dma10[key] = superconductivity.get_eh_sector(m,i=0,j=1)
    mfe = get_mf_normal(v,dme,**kwargs) # electron part of the mean field
    mfa01 = get_mf_anomalous(v,dma10) # anomalous part
    mfa01,mfa10 = enforce_eh_symmetry_anomalous(mfa01)
    mf = dict()
    for key in v:
        if not compute_normal: mfe[key] = mfe[key]*0.0
        if compute_anomalous:
            mf[key] = superconductivity.build_nambu_matrix(mfe[key],
                    c12=mfa10[key],c21=mfa01[key])
        else:
            mf[key] = superconductivity.build_nambu_matrix(mfe[key])
    if not MultiHopping(mf).is_hermitian(): # sanity check on the result
        raise ValueError("Non-Hermitian BdG mean field:\n%s" %
                np.round(mf[(0,0,0)],2))
    return mf # return mean field matrix



def get_charge_bdg(dm):
    """Number of electrons per unit cell held by a Nambu density matrix:
    the trace of the electron block of its onsite matrix dm[(0,0,0)],
    sum_i <c^dag_i c_i>. Every density-matrix integration of the mean
    field ("ed", "triangle", "qtci") computes these diagonal entries for a
    Bogoliubov-de Gennes Hamiltonian, dense or sparse"""
    from ..superconductivity import get_eh_sector
    m = get_eh_sector(dm[(0,0,0)],i=0,j=0)
    return float(np.real(m.diagonal().sum()))



class BdGFermiLevel:
    """The Fermi level of a Bogoliubov-de Gennes mean field at a fixed
    filling, found on the charge of the paired state itself.

    The Fermi level that holds `filling` in the normal state (the
    electron-only Hamiltonian, spectrum.get_fermi4filling) does not hold it
    in the paired state: for U=-2 at filling 0.3 on the square lattice the
    paired state holds 0.606 electrons per cell instead of 0.600. So the
    chemical potential is written as mu = mu_n + x, with mu_n the Fermi
    level of the normal state of the current Hamiltonian (search(h,filling),
    the search of the integration in use) and x an offset that is one more
    unknown of the self-consistent loop. Each density matrix gives the
    charge N of the paired state (get_charge_bdg), and the offset of the
    next iteration is x - gain*damping*(N-N0)/kappa, with N0 the requested
    charge and kappa the compressibility dN/dmu of the normal state at the
    first Hamiltonian. This is one step per iteration and no extra density
    matrix; the loop holds the size of the step (residual) to its tolerance
    together with the change of the mean field, so at convergence a gapped
    paired state holds N0. A mixing solver takes x as one more entry of its
    vector.

    Writing mu around mu_n rather than on its own keeps the uniform
    Hartree shift, which moves mu_n, out of the offset. The offset is also
    clipped to a window |x| <= 2 Delta^3/(Delta^2 + Delta_s^2), with Delta
    the pairing amplitude of the Hamiltonian (the largest entry of its
    anomalous blocks) and Delta_s = small*W, W = (electron states)/kappa an
    energy scale of the bands. So the window is 2 Delta for a sizable
    pairing, and closes as Delta^3 when the pairing dies out: a Hamiltonian
    with no pairing gets exactly the Fermi level of the normal state, as
    before. The gain halves each time the charge error changes sign, and
    grows back by regain up to 1 while it does not. This matters for a
    Nambu Hamiltonian whose state has no gap on the mesh (no pairing, a
    pairing that dies out, a Bogoliubov Fermi surface, or a gap the mesh
    does not resolve): there the charge of "ed" is a staircase in mu and
    the one of the triangles has the kinks of the crossing electron and
    hole bands, so no offset may hold N0. The halving gain lets the offset
    settle at a jump instead of oscillating around it, and the window keeps
    it from running away. When the error has changed sign patience times
    (errors below flip_floor are not counted, they are the noise of the
    interpolation of qtci), the step gives up: the offset is set to zero
    for the rest of the loop, which is then the loop with the Fermi level
    of the normal state, as before. A state with a gap settles in a few
    changes of sign at most. In all these cases the loop converges with a
    charge error, which stays in error, and warn says so.

    Use: shift(h,x) on each Hamiltonian before its density matrix (it sets
    h.fermi), then step(x,dm) for the offset of the next iteration, which
    also leaves the charge error N-N0 in error and the size of the step in
    residual."""
    damping = 0.5 # with kappa off by up to a factor 4 the step still converges
    # the pairing moves the Fermi level by roughly Delta^2/W, so a window of
    # 2 Delta is generous (U=-2 at filling 0.3: Delta=0.21, x=-0.03; U=-6 at
    # filling 0.1: Delta=1.38)
    bound_factor = 2.
    halve = 0.5 # factor of the gain when the error changes sign
    regain = 1.5 # growth of the gain while the error keeps its sign
    small = 1e-3 # a pairing below small*W closes the window
    patience = 8 # changes of sign of the error before giving up
    flip_floor = 1e-6 # smaller errors are noise, and their signs not counted

    def __init__(self,h,filling,search):
        self.filling = filling
        self.search = search # search(h,filling): Fermi level of the normal state
        self.charge0 = filling*(h.intra.shape[0]//2) # electron states per cell
        self.kappa = None # set on the first Hamiltonian
        self.bound = 0. # the window of the last Hamiltonian
        self.x = 0. # the offset the last Hamiltonian was shifted by
        self.error = 0.
        self.residual = 0.
        self.gain, self.last = 1., None
        self.flips, self.frozen = 0, False

    def compressibility(self,h,df=0.05):
        """dN/dmu of the normal state of h around the requested filling,
        from two Fermi searches: a finite difference of the charge over a
        window of 2*df in filling, wide enough to span many levels of an
        exact-diagonalization mesh, whose Fermi level is a staircase"""
        df = min(df,self.filling/2.,(1.-self.filling)/2.)
        if df<=0.: return 1.0 # empty or full, nothing to hold
        dmu = self.search(h,self.filling+df) - self.search(h,self.filling-df)
        dmu = max(dmu,1e-6) # a degenerate level: a small, safe step
        return 2.*df*(h.intra.shape[0]//2)/dmu

    def shift(self,h,x):
        """Shift h by the chemical potential mu_n + x, with x clipped to
        the window of the pairing amplitude of h"""
        mu = self.search(h,self.filling)
        if self.kappa is None: self.kappa = self.compressibility(h)
        d = pairing_amplitude(h)
        dmin = self.small*(h.intra.shape[0]//2)/self.kappa # small, in units of W
        self.bound = self.bound_factor*d*d**2/(d**2+dmin**2)
        self.x = float(np.clip(x,-self.bound,self.bound))
        if self.frozen: self.x = 0. # given up, see step
        h.fermi = mu + self.x
        h.shift_fermi(-h.fermi)
        return h

    def step(self,x,dm):
        """Offset of the next iteration, from the density matrix of the
        Hamiltonian shift(h,x) gave"""
        self.error = get_charge_bdg(dm) - self.charge0
        if self.last is not None: # halve on a change of sign, else grow
            if self.last*self.error<0.:
                self.gain = self.halve*self.gain
                if min(abs(self.last),abs(self.error))>self.flip_floor:
                    self.flips += 1
            else: self.gain = min(1.,self.regain*self.gain)
        self.last = self.error
        if self.flips>=self.patience: self.frozen = True
        if self.frozen: xnew = 0. # the Fermi level of the normal state
        else:
            xnew = self.x - self.gain*self.damping*self.error/self.kappa
            xnew = float(np.clip(xnew,-self.bound,self.bound)) # the window
        self.residual = abs(xnew-x)
        return xnew

    def warn(self,maxerror):
        """Say so when a converged loop does not hold the charge, which is
        the case of a state with no gap on the mesh"""
        if abs(self.error)>100*maxerror:
            print("The paired state holds the requested charge to %.1e "
                    "electrons per cell, not to the tolerance: its "
                    "quasiparticle spectrum has no gap on this mesh, see "
                    "scftk.superscf.BdGFermiLevel"%abs(self.error))



def pairing_amplitude(h):
    """Largest modulus of the anomalous (electron-hole) blocks of the
    hoppings of a Nambu Hamiltonian, onsite and between cells"""
    from ..superconductivity import get_eh_sector
    from .. import algebra
    out = 0.
    for m in h.get_dict().values():
        a = get_eh_sector(algebra.todense(m),i=0,j=1)
        out = max(out,float(np.max(np.abs(a))))
    return out

