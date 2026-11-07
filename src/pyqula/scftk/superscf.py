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



class ShellFilling:
    """The partly filled shell of a gapless Bogoliubov-de Gennes state.

    In a state with no gap on the mesh the paired charge is a staircase in
    mu: when a shell of levels sits at zero energy, the charge jumps by the
    charge of the shell, and the requested charge N0 falls inside the jump,
    so no chemical potential holds it. Here the shell is filled in part
    instead. The shell is the set of BdG levels with |E| < tol, with
    tol = max(width*T, 1e-6/(kappa*Nk)): the levels that the Fermi-Dirac
    weight at T does not turn into a step (width = 40), and at T=0 a floor
    far below the level spacing 1/(kappa*Nk) of the mesh. The jump is
    read on the levels within jump_width*T of zero (jump_width = 5), the
    ones the Fermi-Dirac weight fills in part, by more than 1/150. This is
    where the Fermi search of the normal state puts a degenerate level
    that it fills in part, within T ln(15) < 3 T when one electron or more
    of 16 states is in it, so that the level stays at mu, as the star of
    the Zeeman state does. A state whose Fermi level falls between two
    levels further apart than that has no jump there and keeps its charge
    error.

    The occupations of the shell are the ones of the chemical potential
    mu + y*Ts to first order, so that the whole range where the
    Fermi-Dirac weight is not a step moves with the chemical potential,
    and "ed" and the triangle, which builds its filled density matrix on
    the same occupations, end in the same state. At each k the levels of
    the shell are grouped in degenerate multiplets (within degeneracy,
    1e-12, the rounding of the energies), each multiplet takes the mean of
    its energies (zero for a multiplet at zero within the rounding, so
    that it and its partner at -k have exactly opposite energies), and the
    shell Hamiltonian sum_m E_m P_m - y*Ts*Q is diagonalized, with P_m the
    projector of the multiplet, Q = V^dag tau_z V the charge in the shell
    (tau_z = P_e - P_h) and Ts = tol/width the smearing (T, or the floor
    at T=0). Its eigenvectors get the Fermi-Dirac occupations of their
    eigenvalues at Ts, and y is set so that the trace of the electron
    block, get_charge_bdg(dm), is exactly N0. The shell Hamiltonian is
    built from the projectors, so the filled density matrix depends only
    on the shell, not on the eigenbasis the diagonalization picks inside a
    multiplet. In a multiplet at zero energy, as the star of the Zeeman
    state, the vectors of charge 1 get one occupation f and their
    partners, of charge -1, 1 - f. The partner of a vector at k is at -k
    with the opposite energy and charge, so the filled density matrix
    stays particle-hole symmetric. A shell whose levels are split by more
    than Ts keeps them in the order of their energies. A single f for the
    whole shell does not: on the chain with V1=1 at half filling from a
    ferromagnetic guess (nk=20), whose half filled shell at mu has to
    split, it ended 5e-3 above the normal state. The k-points of the
    shell are handled in groups of the same number of levels, each group
    with one batched diagonalization.

    The shell is armed (start returns True) only when the pairing
    amplitude of the Hamiltonian, pairing (the largest anomalous entry, set
    by BdGFermiLevel.shift), is below the smearing Ts, or when the window
    of the offset, window (also set by shift), is narrower than the shell
    at T=0, floor times the level spacing. This is the case when the
    pairing has died out or was never there, and then the levels of the
    shell are states of one charge, which the first-order shift of the
    chemical potential treats exactly. The second condition does not
    depend on T and is the one that matters at low T: in the Zeeman state
    (U=-2, B=0.3, nk=16) the paired charge jumps while the pairing is
    still about 1e-4, and with a gate on the pairing alone, at 1e-8 times
    the bandwidth, Broyden mixing without its restart went round the gate
    and did not converge in 300 iterations, with the pairing between 7e-8
    and 7e-6. While the pairing lives, the offset of BdGFermiLevel is what
    holds the charge: filling the shell broke the ferromagnetic triplet
    chain of tests/scf/test_unconventional_superconductivity.py (nk=6,
    Delta=1.73), whose level at the node k=1/2 converges to mu, since once
    filled the mean field pushed it 0.18 away and the loop went round a
    cycle of 22 iterations. The gate went through three versions before
    this one. The window alone, which closes as Delta^3, armed live
    states: against the shell, 40 T, the s-wave state with U=-1 at T=1e-2
    (Delta=0.025) in every iteration, and against Ts the one with U=-0.5
    at nk=32 and T=1e-3 (Delta=3.1e-3), whose energy then moved by 6e-7.
    The pairing alone, against Ts or 1e-8 W, let the Zeeman state above go
    round the gate without the restart of Broyden mixing. So the pairing
    is compared with Ts, and the window only with the shell at T=0, which
    is far below any live pairing. A density matrix that is not armed is
    exactly the one of before, on the device route too.

    An armed shell is filled when the density matrix misses N0 by more
    than held electrons per cell, so that a shell that holds its charge
    already, as the triangle of the chain above with its own weights, is
    left alone, and when N0 lies strictly between the charges of the shell
    emptied and filled in charge (y to minus and plus infinity), while
    the charge of the levels within jump_width*Ts, the jump, is larger
    than jump_factor times the charge kappa*2*jump_width*Ts that a smooth
    density of states puts in that window: a smearing that resolves the
    level spacing makes the charge continuous, and then there is no jump
    to fill. test(dm) says whether this holds.

    An armed shell also gives band, the band energy per unit cell of the
    density matrix that the routine returns, (Tr H rho + Tr h_e)/2 with
    the occupations of that density matrix on every level, filled or not.
    The mean field reports it instead of the step band energy, so that the
    energy is the one of the state the loop holds, in one convention on
    both sides of a filling where the fill switches on: dE/dN0 is then
    the chemical potential. It needs only the band energies of the mesh,
    which the density matrix routines pass along with the eigenpairs of
    the shell, so it costs no extra diagonalization.

    The fill moves the chemical potential by mu = y*Ts inside the shell,
    which the mean field adds to the Fermi level of the Hamiltonian it
    returns, shifted by it, so that the Hamiltonian and its density matrix
    describe the same state.

    Use: armed = start(h,T,nkpoints) before a density matrix, and when
    armed, energies(es,occ) on the band energies and occupations of every
    batch, collect(ks,es,vs,occ) on the eigenpairs of those that have
    levels in the shell, and fill(dm) at the end."""
    width = 40. # the shell is |E| < width*T
    jump_width = 5. # the jump is read on |E| < jump_width*T
    floor = 1e-6 # at T=0, the shell in units of the level spacing
    jump_factor = 10. # a jump is this many times the smooth charge
    qzero = 1e-10 # a smaller |q| is a level with no charge
    held = 1e-12 # a charge error per cell below this is held already
    degeneracy = 1e-12 # levels closer than this are one multiplet

    def __init__(self,charge0):
        self.charge0 = charge0 # electrons per cell, as BdGFermiLevel
        self.kappa = None # dN/dmu per cell, set by BdGFermiLevel.shift
        self.pairing = 0. # the pairing amplitude, set by BdGFermiLevel.shift
        self.window = 0. # the window of the offset, set by BdGFermiLevel.shift
        self._reset()

    def _reset(self):
        self.fired, self.fraction, self.mu, self.band = False, None, 0., None
        self.levels, self.groups = [], []
        self.sum_energy = 0. # sum of occupations times band energies

    def start(self,h,T,nkpoints):
        """Before the density matrix of h on a mesh of nkpoints, at T.
        Returns whether the shell is armed, the pairing being dead; if not,
        the density matrix is computed as if there were no shell"""
        from .. import algebra, operators
        self._reset()
        self.nkpoints = nkpoints
        spacing = 1./(self.kappa*nkpoints) # mean level spacing at mu
        self.tol = max(self.width*T,self.floor*spacing)
        self.ts = self.tol/self.width # the smearing of the shell
        if self.pairing>=self.ts and self.window>=self.floor*spacing:
            return False # the pairing lives, the offset holds the charge
        pe = np.array(algebra.todense(operators.get_electron(h))).diagonal()
        self.tau = 2.*pe.real-1. # +1 on the electron components, -1 holes
        intra = np.array(algebra.todense(h.intra)).diagonal().real
        self.trace_e = float(np.sum(pe.real*intra)) # Tr h_e
        return True

    def energies(self,es,occ):
        """Add the band energies es and occupations occ of a batch"""
        self.sum_energy += float(np.sum(occ*es))

    def restart(self):
        """Forget the batches, to collect a density matrix again"""
        self.levels, self.groups, self.sum_energy = [], [], 0.

    def collect(self,ks,es,vs,occ):
        """Keep the shell of a batch: es (nb,n), vs (nb,n,n) with the
        eigenvectors as columns, occ (nb,n) the occupations the density
        matrix was built with, in [0,1]"""
        es = np.asarray(es)
        for ik in np.nonzero(np.any(np.abs(es)<self.tol,axis=1))[0]:
            s = np.abs(es[ik])<self.tol
            self.levels.append((np.array(ks[ik],dtype=float),
                np.array(es[ik][s]),vs[ik][:,s],np.array(occ[ik][s])))

    def _multiplets(self,es):
        """es (g,m), sorted along the rows, replaced by the mean of their
        multiplet, and by zero within the rounding of it"""
        g,m = es.shape
        cut = np.diff(es,axis=1)>=self.degeneracy # a new multiplet
        label = np.concatenate([np.zeros((g,1),dtype=int),
            np.cumsum(cut,axis=1)],axis=1)+m*np.arange(g)[:,None]
        total = np.bincount(label.ravel(),weights=es.ravel(),minlength=g*m)
        count = np.bincount(label.ravel(),minlength=g*m)
        out = (total/np.maximum(count,1))[label]
        out[np.abs(out)<self.degeneracy] = 0.
        return out

    def _group(self):
        """the levels of the shell in groups of k-points with the same
        number of levels: ks (g,3), es, em (g,m), vs (g,n,m), occ (g,m),
        Q (g,m,m)"""
        self.groups = []
        sizes = np.array([len(l[1]) for l in self.levels])
        for m in np.unique(sizes):
            ls = [self.levels[i] for i in np.nonzero(sizes==m)[0]]
            ks = np.array([l[0] for l in ls])
            es = np.array([l[1] for l in ls])
            vs = np.array([l[2] for l in ls])
            occ = np.array([l[3] for l in ls])
            Q = np.einsum("gim,i,gin->gmn",vs.conj(),self.tau,vs)
            self.groups.append((ks,es,self._multiplets(es),vs,occ,Q))

    def _charges(self,q):
        """electron weights of the levels of negative and positive charge"""
        w = (1.+q)/2.
        return (np.sum(w*(q<-self.qzero)),np.sum(w*(q>self.qzero)))

    def test(self,dm):
        """Whether fill(dm) fills the shell"""
        from scipy.special import expit
        if len(self.levels)==0: return False # no level at zero energy
        n = get_charge_bdg(dm)
        if abs(n-self.charge0)<=self.held: return False # held already
        nk = float(self.nkpoints)
        self._group()
        old, c0, c1, jump = 0., 0., 0., 0.
        for (ks,es,em,vs,occ,Q) in self.groups:
            old += np.sum(occ*np.real(np.einsum("gmm->gm",Q)+1.)/2.)/nk
            q,R = np.linalg.eigh(Q)
            a,b = self._charges(q)
            c0 += a/nk # emptied in charge, the vectors of q<0 are filled
            c1 += b/nk # and filled in charge, the vectors of q>0
            mask = (np.abs(es)<self.jump_width*self.ts).astype(float)
            a,b = self._charges(np.linalg.eigvalsh(
                Q*mask[:,:,None]*mask[:,None,:])) # the levels at mu
            jump += (b-a)/nk
            for ig in np.nonzero(np.any(np.abs(q)<=self.qzero,axis=1))[0]:
                z = np.abs(q[ig])<=self.qzero # no charge: the occupation
                Rz = R[ig][:,z] # of their energies at any y
                e = np.linalg.eigvalsh(Rz.conj().T@(em[ig][:,None]*Rz))
                c = np.sum(expit(-e/self.ts)*(1.+q[ig][z])/2.)/nk
                c0, c1 = c0+c, c1+c
        self.c0, self.c1, self.jump = c0+n-old, c1+n-old, jump
        smooth = self.kappa*2.*self.jump_width*self.ts # no jump at mu
        return bool(jump>self.jump_factor*smooth and
                self.c0<self.charge0<self.c1)

    def _occupations(self,y):
        """Rotations, occupations and charge per cell of the shell at y"""
        from scipy.special import expit
        out, charge = [], 0.
        for (ks,es,em,vs,occ,Q) in self.groups:
            m = es.shape[1]
            lam,R = np.linalg.eigh(em[:,:,None]*np.eye(m)/self.ts-y*Q)
            on = expit(-lam) # Fermi-Dirac at the smearing ts
            q = np.real(np.einsum("gmi,gmn,gni->gi",R.conj(),Q,R))
            charge += np.sum(on*(1.+q)/2.)/self.nkpoints
            out.append((R,on))
        return out,charge

    def _band(self,change=0.):
        """the band energy of the density matrix, with the change of the
        sum of occupations times energies made by the fill"""
        self.band = (self.sum_energy/self.nkpoints+change+self.trace_e)/2.

    def fill(self,dm):
        """Fill the shell in part so that dm holds charge0, if test(dm)
        passes; dm is changed in place and returned. Sets band either way"""
        if not self.test(dm):
            self._band()
            return dm
        from scipy.optimize import brentq
        nk = float(self.nkpoints)
        def error(y): return self._occupations(y)[1]+self.c0-self.charge0
        b = 2.*self.width # every level of the shell is within width*ts
        while error(-b)>=0. or error(b)<=0.: # q below 1 needs more
            b *= 4.
            if b>1e12: # the charge of the shell is too small
                self._band()
                return dm
        y = brentq(error,-b,b,xtol=1e-14,rtol=1e-14,maxiter=500)
        change = 0.
        for ((ks,es,em,vs,occ,Q),(R,on)) in zip(self.groups,
                self._occupations(y)[0]):
            ws = np.einsum("gim,gmn->gin",vs,R) # eigenvectors at y
            change += np.sum(on*np.einsum("gm,gmn->gn",es,np.abs(R)**2))
            change -= np.sum(occ*es)
            for d in dm:
                ph = np.exp(2j*np.pi*(ks@np.array(d,dtype=float)))/nk
                new = np.einsum("gin,gn,gjn->ij",ws.conj(),on*ph[:,None],ws)
                old = np.einsum("gin,gn,gjn->ij",vs.conj(),occ*ph[:,None],vs)
                dm[d] = dm[d]+new-old
        self._band(change/nk)
        self.fired, self.mu = True, y*self.ts
        self.fraction = (self.charge0-self.c0)/(self.c1-self.c0) # filled
        return dm

    def move_fermi(self,h):
        """Add the shift of the chemical potential of the last fill to the
        Fermi level of h, shifting h by it, so that h and the filled
        density matrix describe the same state; nothing when not filled"""
        if not self.fired: return h
        h.shift_fermi(-self.mu)
        h.fermi = h.fermi + self.mu
        return h



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
    before. The gain halves each time the charge error changes sign, down
    to gain_min, and grows back by regain up to 1 while it does not. This
    matters for a Nambu Hamiltonian whose state has no gap on the mesh (no
    pairing, a pairing that dies out, a Bogoliubov Fermi surface, or a gap
    the mesh does not resolve): there the charge is a staircase in mu (the
    triangle density matrix of a BdG Hamiltonian is the step one of exact
    diagonalization, see dmtk/simplexdm.py), so the offset may have to sit
    on a level, where the smearing T makes the charge continuous, or no
    offset may hold N0 at all. The halving gain lets the offset settle on a
    level instead of oscillating around it. The step gives up, and sets
    the offset to zero for the rest of the loop, which is then the loop
    with the Fermi level of the normal state as before, in two cases: when
    the window has held the step patience times in a row, and when the
    error has changed sign flip_patience times at the smallest gain (errors
    below flip_floor are not counted, they are the noise of the
    interpolation of qtci). A state with a gap settles long before. When
    the step gives up the loop converges with a charge error, which stays
    in error, and warn says so.

    The halving gain, the damping and the give-up are a history of the
    loop, so they belong to the plain loop only. A quasi-Newton solver
    (Broyden mixing, the scipy solvers) takes the offset as one more entry
    of its vector and fits a model of the map from that vector to the next
    one, which only works if the map is a function of the vector. With the
    gain inside, the same trial vector gave a different step at each call:
    for a d-wave state from V1=-2 with U=1 at filling 0.4 on the square
    lattice the gain moved between 1 and 0.18 in one Broyden run, which
    then needed 48 iterations. So those solvers call step with
    adaptive=False, a step that depends only on the Hamiltonian and the
    offset: the Newton step -(N-N0)/kappa, clipped to the window, and zero
    when it is longer than the whole window, 2*bound, since then no offset
    in the window holds the charge on the linear estimate. This last rule
    replaces the give-up for the solvers. In a state whose pairing dies
    out, the charge error is a staircase of order one k-shell while the
    window closes as Delta^3, so the step soon becomes longer than the
    window and the offset stays at zero, the Fermi level of the normal
    state, which is where the give-up of the plain loop ends too. The
    d-wave state above needs 20 iterations with it.

    No offset holds N0 in a state with no gap on the mesh, since N0 falls
    inside a jump of the staircase. That jump is the shell of levels at
    zero energy, and the density matrices of "ed" and "triangle" fill it
    in part when they are given shell, the ShellFilling of this object:
    the shell is occupied at a chemical potential shifted inside it, set
    so that the charge is exactly N0, which in a shell degenerate at zero
    energy gives the occupations f and 1-f to its two charges (see
    ShellFilling). It fires only when the density matrix misses N0 and
    the charge jumps across N0 there, so a gapped state or one that holds
    its charge already does not change, and only once the pairing is
    below the smearing of the shell, or its window below the shell at
    T=0, so that while the pairing lives the offset alone holds the
    charge. With it the Zeeman state below (U=-2 at
    filling 0.3 with a field 0.3 that kills the pairing) holds N0 to 1e-16
    instead of ending with the error of one k-shell, 1.6e-3 per cell at
    nk=16, and "ed" and the triangle give the same energy and Fermi level.
    The step itself is unchanged: there N - N0 is zero, so the offset
    stays where it is.

    Use: shift(h,x) on each Hamiltonian before its density matrix (it sets
    h.fermi), then step(x,dm) for the offset of the next iteration, which
    also leaves the charge error N-N0 in error and the size of the step in
    residual. Pass shell to the density matrix; when shell.band is not
    None (the shell was armed), report it as the band energy of the last
    Hamiltonian, and shift that Hamiltonian with shell.move_fermi."""
    damping = 0.5 # with kappa off by up to a factor 4 the step still converges
    # the pairing moves the Fermi level by roughly Delta^2/W, so a window of
    # 2 Delta is generous (U=-2 at filling 0.3: Delta=0.21, x=-0.03; U=-6 at
    # filling 0.1: Delta=1.38)
    bound_factor = 2.
    halve = 0.5 # factor of the gain when the error changes sign
    regain = 1.5 # growth of the gain while the error keeps its sign
    small = 1e-3 # a pairing below small*W closes the window
    patience = 8 # steps in a row held by the window, to give up
    flip_patience = 8 # changes of sign at the smallest gain, to give up
    gain_min = 1./64. # the smallest gain
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
        self.flips, self.clipped, self.frozen = 0, 0, False
        # the partly filled shell of a gapless state, which the density
        # matrices of "ed" and "triangle" take as shell=
        self.shell = ShellFilling(self.charge0)

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
        if self.kappa is None:
            self.kappa = self.compressibility(h)
            self.shell.kappa = self.kappa
        d = pairing_amplitude(h)
        dmin = self.small*(h.intra.shape[0]//2)/self.kappa # small, in units of W
        self.bound = self.bound_factor*d*d**2/(d**2+dmin**2)
        self.shell.pairing = d # the shell waits for the pairing to die
        self.shell.window = self.bound
        self.x = float(np.clip(x,-self.bound,self.bound))
        if self.frozen: self.x = 0. # given up, see step
        h.fermi = mu + self.x
        h.shift_fermi(-h.fermi)
        return h

    def step(self,x,dm,adaptive=True):
        """Offset of the next iteration, from the density matrix of the
        Hamiltonian shift(h,x) gave. adaptive=False is the step of the
        quasi-Newton solvers, which depends only on x and dm: no gain, no
        damping and no give-up (see the class docstring)"""
        self.error = get_charge_bdg(dm) - self.charge0
        if not adaptive: # the map of x and dm alone, for a solver
            dx = -self.error/self.kappa # the Newton step of the offset
            # longer than the window: no offset in it holds the charge,
            # wherever x is, so the Fermi level of the normal state
            if abs(dx)>2.*self.bound: xnew = 0.
            else: xnew = float(np.clip(self.x+dx,-self.bound,self.bound))
            self.residual = abs(xnew-x)
            return xnew
        if self.last is not None: # halve on a change of sign, else grow
            if self.last*self.error<0.:
                # a change of sign at the smallest gain is no progress
                if self.gain<=self.gain_min and \
                        min(abs(self.last),abs(self.error))>self.flip_floor:
                    self.flips += 1
                self.gain = max(self.gain_min,self.halve*self.gain)
            else: self.gain = min(1.,self.regain*self.gain)
        self.last = self.error
        if self.flips>=self.flip_patience: self.frozen = True
        if self.frozen: xnew = 0. # the Fermi level of the normal state
        else:
            xnew = self.x - self.gain*self.damping*self.error/self.kappa
            if abs(xnew)>self.bound: self.clipped += 1 # held by the window
            else: self.clipped = 0
            xnew = float(np.clip(xnew,-self.bound,self.bound)) # the window
            if self.clipped>=self.patience: # the charge is out of reach
                self.frozen, xnew = True, 0.
        self.residual = abs(xnew-x)
        return xnew

    def warn(self,maxerror,converged=True):
        """Say so when a converged loop does not hold the charge, which is
        the case of a state with no gap on the mesh. A loop stopped at
        maxite says nothing about the gap, so then nothing is printed"""
        if not converged: return
        if abs(self.error)>100*maxerror:
            print("The paired state holds the requested charge to %.1e "
                    "electrons per cell, not to the tolerance: its "
                    "quasiparticle spectrum has no gap on this mesh, see "
                    "scftk.superscf.BdGFermiLevel"%abs(self.error))



def pairing_amplitude(h):
    """Largest modulus of the anomalous (electron-hole) blocks of the
    hoppings of a Nambu Hamiltonian, onsite and between cells. The Nambu
    spinor of site i is (4i, 4i+1) for the electrons and (4i+2, 4i+3) for
    the holes, so the anomalous block is read with these indices, without
    the reordering of superconductivity.get_eh_sector, which densified and
    permuted every hopping at each call (12% of an iteration of the s-wave
    loop at nk=16, a quarter of BdGFermiLevel.shift)"""
    from scipy.sparse import issparse
    i = np.arange(h.intra.shape[0]//4) # Nambu sites
    e = np.concatenate([4*i,4*i+1]) # electrons
    o = np.concatenate([4*i+2,4*i+3]) # holes
    out = 0.
    for m in h.get_dict().values():
        a = m[e,:][:,o]
        a = a.toarray() if issparse(a) else np.asarray(a)
        out = max(out,float(np.max(np.abs(a))))
    return out
