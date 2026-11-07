# Prototype of a basis per simplex for the bands that cross inside a
# simplex of the triangle susceptibility, chiAB_q(..., integration=
# "triangle"), whose engine is chitk/simplexchi.py. Nothing here is in the
# library; the design and the numbers are in tetrahedron_susceptibility.md,
# section "A basis per simplex for crossing bands".
#
# The library interpolates E_n and G_nm = <n,k|A|m,k+q><m,k+q|B|n,k> per
# vertex, with the bands sorted by energy at each vertex, and averages G
# over the degenerate multiplets of each vertex. Here, per simplex and per
# side (the bands at k and those at k+q):
#  1. at a vertex where a multiplet is degenerate but not on the whole
#     simplex (a crossing on the vertex), the basis of the multiplet is
#     fixed by first-order degenerate perturbation theory along a direction
#     into the simplex, the eigenvectors of P (dH/dk . d) P (rotation="kp"),
#     or by the largest overlap with the reference vertex (rotation=
#     "polar"). What stays degenerate after that is aligned by the polar
#     decomposition of its overlap with the reference vertex;
#  2. the bands of each cluster (the bands whose energy ranges on the
#     simplex come within window times the largest range of a band) are
#     relabeled at every vertex by the largest overlap with the reference
#     vertex, the vertex with the most distinct levels;
#  3. the relabeled energies are interpolated linearly, so the exact
#     kernel of the library applies unchanged, on private copies of the
#     vertices of every simplex that changed (virtual vertices), and the
#     matrix elements are those of the relabeled eigenvectors. The other
#     simplices keep the shared vertices and the multiplet average.
# Since the basis at every vertex is an eigenbasis, the 2x2 block of the
# Hamiltonian in it is diagonal at every vertex, and the eigenvalues of
# its linear interpolation are the linearly interpolated relabeled
# energies.
#
# Usage: python chi_simplex_basis.py <task> <arguments>, with the tasks at
# the end (check, direction, reference, convergence, static, cost). It
# needs pyqula on the path and writes its .npy files to the working
# directory. Spinless 2x2 honeycomb supercell, 8 bands, takes seconds to
# minutes per call on three cores; the grid references at nk = 1536 take
# about 15 minutes each.
import sys
import time
import json
import numpy as np
from numba import jit, prange
import numba

from pyqula import geometry, parallel
from pyqula.chitk import simplexchi as sc
from pyqula.chitk import chiAB
from pyqula.dostk import simplexdos as sd
from pyqula.htk.eigenvectors import parallel_diagonalization


# ------------------------------------------------------------ models

def supercell(spin=False,onsite=0.3,modulation=0.,random=0.):
    """the 2x2 honeycomb supercell with a uniform onsite energy, whose
    folded bands cross on lines; modulation adds an onsite energy to the
    first site only, which breaks the primitive translations and opens the
    crossings, and random adds a random onsite energy of that size to
    every site, which leaves no degeneracy on the mesh"""
    g = geometry.honeycomb_lattice().get_supercell(2)
    h = g.get_hamiltonian(has_spin=spin)
    h.add_onsite(onsite)
    if modulation!=0. or random!=0.:
        if h.is_sparse: h = h.get_dense()
        ns = len(g.r)
        d = np.zeros(ns); d[0] = modulation
        d += random*np.random.default_rng(1).uniform(-1.,1.,ns)
        if spin: d = np.repeat(d,2)
        h.intra = h.intra + np.diag(d)
    return h


def projectors(h):
    """the site projectors of chiAB_q, as dense matrices"""
    from pyqula import operators, algebra
    n = h.intra.shape[0]
    A = np.identity(n,dtype=np.complex128)
    return np.array([np.asarray(algebra.todense(operators.index(h,n=[i])@A),
            dtype=np.complex128) for i in range(len(h.geometry.r))])


# ------------------------------------------------------------ library calls

def chi_library(h,q,nk,energies,delta,T,P,kernel=None):
    return sc.chi_triangle(h,q=q,energies=energies,nk=nk,pAs=P,pBs=P,
            delta=delta,T=T,mode="matrix",kernel=kernel)[1]


def chi_grid(h,q,nk,energies,delta,T):
    return np.array(chiAB.chiAB_q(h,q=q,energies=energies,nk=nk,delta=delta,
            T=T,mode="matrix")[1])


# ------------------------------------------------------------ the transport

def eig(hkgen,ks,batch=4096):
    E = []; U = []
    for i0 in range(0,len(ks),batch):
        e,u = parallel_diagonalization(sd._hk_batch(hkgen,ks[i0:i0+batch]))
        E.append(np.array(e)); U.append(np.array(u))
    return np.concatenate(E),np.ascontiguousarray(np.concatenate(U))


def dH_batch(hkgen,ks,dirs):
    """dH/dk . d at every k, from the Bloch data of the generator"""
    ms,ds = hkgen.bloch_data
    dim = ds.shape[1]
    out = np.zeros((len(ks),ms.shape[1],ms.shape[2]),dtype=np.complex128)
    for i0 in range(0,len(ks),4096):
        k = ks[i0:i0+4096,:dim]; d = dirs[i0:i0+4096,:dim]
        f = 2j*np.pi*(d@ds.T)*np.exp(2j*np.pi*(k@ds.T))
        out[i0:i0+len(k)] = np.einsum("kh,hij->kij",f,ms,optimize=True)
    return out


@jit(nopython=True,cache=True)
def _deg_vertices(sim,E,tol,item):
    """item[s,v] = 1 where the vertex v of the simplex s has a degenerate
    pair of bands that is not degenerate on the whole simplex"""
    ns,p = sim.shape; nb = E.shape[1]
    cnt = 0
    for s in range(ns):
        for v in range(p):
            k = sim[s,v]
            for n in range(nb-1):
                if E[k,n+1]-E[k,n]<tol:
                    whole = True
                    for j in range(p):
                        if E[sim[s,j],n+1]-E[sim[s,j],n]>=tol: whole = False
                    if not whole:
                        item[s,v] = 1; cnt += 1
                        break
    return cnt


def directions(sim,ks):
    """a generic direction into the simplex from each of its vertices, a
    combination with positive weights of the edges to the other vertices
    (the mesh is periodic, so the edges are unwrapped)"""
    ns,p = sim.shape
    w = {2:[1.],3:[0.6180339887,0.3819660113],
         4:[0.5402,0.2918,0.1680]}[p]
    d = np.zeros((ns,p,3))
    for v in range(p):
        others = [j for j in range(p) if j!=v]
        for c,j in enumerate(others):
            e = ks[sim[:,j]]-ks[sim[:,v]]
            e = (e+0.5)%1.-0.5
            d[:,v] += w[c]*e
    return d


@jit(nopython=True,cache=True)
def _groups(e,n0,n1,tol,gs):
    """the start of every group of consecutive values of e[n0:n1] closer
    than tol, written to gs; returns their number (gs[ng] = n1)"""
    ng = 0
    gs[0] = n0
    for n in range(n0+1,n1):
        if e[n]-e[n-1]>=tol:
            ng += 1; gs[ng] = n
    ng += 1
    gs[ng] = n1
    return ng


@jit(nopython=True,cache=True)
def _polar(B,T):
    """the unitary R that makes B R closest to T, B and T with the same
    number of orthonormal columns: R = X Y^dag for B^dag T = X S Y^dag"""
    M = np.conj(B.T)@T
    X,S,Yh = np.linalg.svd(M)
    return X@Yh


@jit(nopython=True,parallel=True,cache=True)
def _transport(sim,E,U,tol,rotation,window,item_idx,dHs,Ev,Vv,flag,stats):
    """the relabeled energies Ev[s,v,a] and eigenvectors Vv[s,v,:,a] of
    every simplex s, and flag[s] = 1 where they differ from the shared
    vertices. rotation: 0 none (the plain sorted bands, every simplex
    flagged, for the check against the library), 1 polar, 2 kp.
    stats[s,0] counts the relabeled bands, stats[s,1] the multiplets that
    stay degenerate after the kp rotation"""
    ns,p = sim.shape; nb = E.shape[1]; no = U.shape[1]
    for s in prange(ns):
        Vl = np.zeros((p,no,nb),dtype=np.complex128)
        El = np.zeros((p,nb))
        for v in range(p):
            Vl[v] = U[sim[s,v]]
            El[v] = E[sim[s,v]]
        if rotation==0:
            Ev[s] = El; Vv[s] = Vl; flag[s] = 1
            continue
        changed = False
        # group boundaries per vertex: gstart[v,0..ng[v]]
        gstart = np.zeros((p,nb+1),dtype=np.int64)
        ngv = np.zeros(p,dtype=np.int64)
        gs = np.zeros(nb+1,dtype=np.int64)
        for v in range(p):
            ng = _groups(El[v],0,nb,tol,gs)
            if item_idx[s,v]>=0: # a crossing on this vertex
                changed = True
                if rotation==2: # kp: split the multiplets with P dH P
                    dH = dHs[item_idx[s,v]]
                    ng2 = 0
                    gs2 = np.zeros(nb+1,dtype=np.int64)
                    for g in range(ng):
                        a0 = gs[g]; a1 = gs[g+1]
                        if a1-a0==1:
                            gstart[v,ng2] = a0; ng2 += 1
                            continue
                        B = np.ascontiguousarray(Vl[v][:,a0:a1])
                        M = np.conj(B.T)@(dH@B)
                        M = 0.5*(M+np.conj(M.T))
                        w,R = np.linalg.eigh(M)
                        Vl[v][:,a0:a1] = B@R
                        sc_ = 0.
                        for i in range(a1-a0): sc_ = max(sc_,abs(w[i]))
                        tk = 1e-6*sc_+1e-10
                        nsub = _groups(w,0,a1-a0,tk,gs2)
                        for i in range(nsub):
                            gstart[v,ng2] = a0+gs2[i]; ng2 += 1
                            if gs2[i+1]-gs2[i]>1: stats[s,1] += 1
                    gstart[v,ng2] = nb
                    ngv[v] = ng2
                    continue
            for g in range(ng+1): gstart[v,g] = gs[g]
            ngv[v] = ng
        # the reference vertex: the most distinct levels, counted with a
        # tolerance of 1e-3 of the largest range of a band on the simplex,
        # so that the choice does not jump when a crossing on a vertex
        # splits by a tiny amount (a q next to a mesh q)
        spr = 0.
        for n in range(nb):
            a = El[0,n]; b = El[0,n]
            for v in range(1,p):
                a = min(a,El[v,n]); b = max(b,El[v,n])
            spr = max(spr,b-a)
        r = 0; best = -1
        for v in range(p):
            nl = _groups(El[v],0,nb,1e-3*spr+tol,gs)
            if nl>best:
                best = nl; r = v
        # clusters: consecutive bands whose ranges on the simplex come
        # closer than window times the largest range of a band. Two bands
        # that cross inside the simplex can have disjoint sorted ranges, by
        # up to the spread of their difference, so the window is not tol
        lo = np.zeros(nb); hi = np.zeros(nb)
        spread = 0.
        for n in range(nb):
            lo[n] = El[0,n]; hi[n] = El[0,n]
            for v in range(1,p):
                lo[n] = min(lo[n],El[v,n]); hi[n] = max(hi[n],El[v,n])
            spread = max(spread,hi[n]-lo[n])
        wtol = window*spread+tol
        Evs = El.copy()
        Vvs = Vl.copy()
        c0 = 0
        while c0<nb:
            c1 = c0+1
            hmax = hi[c0]
            while c1<nb and lo[c1]<=hmax+wtol:
                hmax = max(hmax,hi[c1]); c1 += 1
            nc = c1-c0
            if nc>1:
                Tr = np.ascontiguousarray(Vl[r][:,c0:c1])
                for v in range(p):
                    if v==r: continue
                    # the groups of v inside the cluster
                    glist0 = np.zeros(nc,dtype=np.int64)
                    glist1 = np.zeros(nc,dtype=np.int64)
                    ngc = 0
                    for g in range(ngv[v]):
                        if gstart[v,g]>=c0 and gstart[v,g]<c1:
                            glist0[ngc] = gstart[v,g]
                            glist1[ngc] = min(gstart[v,g+1],c1)
                            ngc += 1
                    B = np.ascontiguousarray(Vl[v][:,c0:c1])
                    O = np.conj(B.T)@Tr # <b at v|a at r>
                    W = np.zeros((ngc,nc))
                    for g in range(ngc):
                        for b in range(glist0[g],glist1[g]):
                            for a in range(nc): W[g,a] += abs(O[b-c0,a])**2
                    room = glist1[:ngc]-glist0[:ngc]
                    owner = -np.ones(nc,dtype=np.int64)
                    for it in range(nc): # greedy assignment
                        best = -1.; bg = -1; ba = -1
                        for g in range(ngc):
                            if room[g]==0: continue
                            for a in range(nc):
                                if owner[a]>=0: continue
                                if W[g,a]>best:
                                    best = W[g,a]; bg = g; ba = a
                        owner[ba] = bg; room[bg] -= 1
                    for g in range(ngc):
                        d = glist1[g]-glist0[g]
                        labs = np.zeros(d,dtype=np.int64)
                        j = 0
                        for a in range(nc):
                            if owner[a]==g:
                                labs[j] = a; j += 1
                        Bg = np.ascontiguousarray(Vl[v][:,glist0[g]:glist1[g]])
                        if d>1:
                            Tg = np.zeros((no,d),dtype=np.complex128)
                            for i in range(d): Tg[:,i] = Tr[:,labs[i]]
                            Bg = Bg@_polar(Bg,Tg)
                        for i in range(d):
                            a = labs[i]
                            Vvs[v][:,c0+a] = Bg[:,i]
                            Evs[v,c0+a] = El[v,glist0[g]+i]
                            if c0+a!=glist0[g]+i:
                                changed = True; stats[s,0] += 1
            c0 = c1
        Ev[s] = Evs; Vv[s] = Vvs
        if changed: flag[s] = 1


def transport_side(sim,ks,E,U,hkgen,tol,rotation,window=2.):
    ns,p = sim.shape
    nb = E.shape[1]; no = U.shape[1]
    item = np.zeros((ns,p),dtype=np.int64)
    _deg_vertices(sim,E,tol,item)
    item_idx = -np.ones((ns,p),dtype=np.int64)
    sel = np.nonzero(item.ravel())[0]
    item_idx.ravel()[sel] = np.arange(len(sel))
    if rotation==2 and len(sel)>0:
        d = directions(sim,ks).reshape(ns*p,3)[sel]
        kv = ks[sim.ravel()[sel]]
        dHs = dH_batch(hkgen,kv,d)
    else: dHs = np.zeros((1,no,no),dtype=np.complex128)
    Ev = np.zeros((ns,p,nb)); Vv = np.zeros((ns,p,no,nb),dtype=np.complex128)
    flag = np.zeros(ns,dtype=np.int64)
    stats = np.zeros((ns,2),dtype=np.int64)
    _transport(sim,E,U,tol,rotation,window,item_idx,dHs,Ev,Vv,flag,stats)
    return Ev,Vv,flag,{"degenerate_vertices":int(len(sel)),
            "relabeled":int(stats[:,0].sum()),
            "kp_still_degenerate":int(stats[:,1].sum())}


# ------------------------------------------------------------ the response

def elements(U1,U2,P,ia,ib,batch=2048):
    """G[k,n,m,c] = <n,k|P_ia|m,k+q><m,k+q|P_ib|n,k> as in
    simplexchi._chi_sources"""
    nv,no,nb = U1.shape
    G = np.zeros((nv,nb,nb,len(ia)),dtype=np.complex128)
    for i0 in range(0,nv,batch):
        u1 = U1[i0:i0+batch]; u2 = U2[i0:i0+batch]
        MA = np.conj(np.transpose(u1,(0,2,1)))[:,None]@(P[None]@u2[:,None])
        MB = np.conj(np.transpose(u2,(0,2,1)))[:,None]@(P[None]@u1[:,None])
        MB = np.transpose(MB,(0,1,3,2))
        G[i0:i0+len(u1)] = np.transpose(MA[:,ia]*MB[:,ib],(0,2,3,1))
    return G


def response_sources_sim(sim,frac,E1,E2,G,omegas,delta,T):
    """simplexchi.response_sources with the simplices given"""
    from pyqula.dostk.simplexdos import _degeneracy_tol
    E1 = np.ascontiguousarray(E1,dtype=np.float64)
    E2 = np.ascontiguousarray(E2,dtype=np.float64)
    G = np.ascontiguousarray(G,dtype=np.complex128)
    omegas = np.ascontiguousarray(omegas,dtype=np.float64)
    tol = _degeneracy_tol(np.concatenate([E1,E2]))
    ns,p = sim.shape
    nw = len(omegas); nc = G.shape[3]
    delta = float(delta); T = float(T)
    dmin = float(np.min(E1)-np.max(E2)); dmax = float(np.max(E1)-np.min(E2))
    span = max(dmax-dmin,1e-8*max(1.,abs(dmin),abs(dmax)))
    off,binc,binw = sc._bins(dmin,span)
    nbins = len(binc)
    fk = np.array([np.prod(np.arange(1.,p))*np.prod(np.arange(1.,k+1))
            /np.prod(np.arange(1.,k+p+1)) for k in range(sc._nmom)])
    mom = np.zeros((nbins,sc._nmom,nc),dtype=np.complex128)
    occupied = np.zeros(nbins,dtype=np.bool_)
    parallel.set_num_threads()
    out = np.zeros((nw,nc),dtype=np.complex128)
    dgsum = np.zeros(nc,dtype=np.complex128)
    resonant = T>0. and delta<T
    nfix = sc._maxblk
    if resonant: nfix = max(1,min(sc._maxblk,int(2e8)//(nw*nc*16)))
    fix = np.zeros((nfix,nw,nc) if resonant else (nfix,0,nc),
            dtype=np.complex128)
    maxsrc = max(1024,min(sc._maxsrc,int(2e8)//(p*(8+16*nc)+8)))
    for s0 in range(0,ns,sc._maxsim):
        s1 = min(ns,s0+sc._maxsim)
        bound = np.zeros(s1-s0,dtype=np.int64)
        sc._bounds_jit(sim,s0,s1,E1,E2,T,tol,sc._tbreaks,bound)
        cum = np.cumsum(bound)
        i0 = 0
        while i0<len(bound):
            base = cum[i0-1] if i0>0 else 0
            i1 = int(np.searchsorted(cum,base+maxsrc,side="right"))
            i1 = max(i1,i0+1)
            cnt = bound[i0:i1]
            npc = int(np.sum(cnt))
            offs = np.concatenate([[0],np.cumsum(cnt)[:-1]]).astype(np.int64)
            pt = np.empty((npc,p)); pc = np.empty((npc,p,nc),dtype=np.complex128)
            pb = np.full(npc,-1,dtype=np.int64)
            dg = np.zeros((i1-i0,nc),dtype=np.complex128)
            sc._pieces_jit(sim,s0+i0,s0+i1,E1,E2,G,T,tol,sc._tbreaks,offs,pt,
                    pc,pb,dg,dmin,span,off,omegas,delta,fix)
            dgsum += np.sum(dg,axis=0)
            order,starts = sc._sort_by_bin(pb,nbins)
            npc = len(order)
            if npc>0:
                occ = np.nonzero(starts[1:]>starts[:-1])[0].astype(np.int64)
                occupied[occ] = True
                cnt = np.cumsum(starts[occ+1]-starts[occ])
                ntask = min(len(occ),16*numba.get_num_threads())
                tasks = np.searchsorted(cnt,np.arange(ntask+1)*(cnt[-1]/ntask),
                        side="left")
                tasks[0] = 0; tasks[-1] = len(occ)
                tasks = np.maximum.accumulate(tasks).astype(np.int64)
                sc._moments_jit(order,starts,occ,tasks,pt,pc,binc,binw,fk,mom)
                nchunk = max(1,min(npc//256,4*numba.get_num_threads()))
                part = np.zeros((nw,nchunk,nc),dtype=np.complex128)
                sc._near_jit(order,starts,pt,pc,binc,binw,off,dmin,span,omegas,
                        delta,nchunk,part)
                out += np.sum(part,axis=1)
            i0 = i1
    sc._far_jit(np.nonzero(occupied)[0].astype(np.int64),mom,binc,binw,omegas,
            delta,out)
    out += dgsum[None,:]*sc._static_factors(omegas,delta)[:,None]
    if resonant: out += np.sum(fix,axis=0)
    return out*frac


def chi_proto(h,q,nk,energies,delta,T,P,rotation=2,info=None,window=2.):
    """the response with the basis per simplex, mode="matrix" of chiAB_q,
    (nw, ni, nj). rotation: 0 plain sorted bands per simplex (no multiplet
    average, the check of the plumbing), 1 polar, 2 kp"""
    t = {}
    t0 = time.perf_counter()
    if h.is_sparse: h = h.get_dense()
    dim = h.dimensionality
    hkgen = h.get_hk_gen()
    ks = sd.uniform_mesh(dim,nk)
    qv = np.zeros(3); qv[:len(q)] = q
    kq = ks+qv[None,:]
    E1,U1 = eig(hkgen,ks); E2,U2 = eig(hkgen,kq)
    t["diagonalization"] = time.perf_counter()-t0
    sim,frac = sd.simplices(dim,nk)
    tol = sd._degeneracy_tol(np.concatenate([E1,E2]))
    t0 = time.perf_counter()
    Ev1,Vv1,f1,s1 = transport_side(sim,ks,E1,U1,hkgen,tol,rotation,window)
    Ev2,Vv2,f2,s2 = transport_side(sim,kq,E2,U2,hkgen,tol,rotation,window)
    flag = np.nonzero((f1+f2)>0)[0]
    t["transport"] = time.perf_counter()-t0
    ns,p = sim.shape; nb = E1.shape[1]
    nkk = len(ks)
    ni = len(P)
    comps = [(i,j) for i in range(ni) for j in range(ni)]
    ia = np.array([c[0] for c in comps]); ib = np.array([c[1] for c in comps])
    nf = len(flag)
    simall = sim.copy()
    simall[flag] = nkk+np.arange(nf*p).reshape(nf,p)
    E1all = np.concatenate([E1,Ev1[flag].reshape(nf*p,nb)])
    E2all = np.concatenate([E2,Ev2[flag].reshape(nf*p,nb)])
    V1f = Vv1[flag].reshape(nf*p,*Vv1.shape[2:])
    V2f = Vv2[flag].reshape(nf*p,*Vv2.shape[2:])
    nvert = nkk+nf*p
    cblock = max(1,int(sc.max_memory_gb*1e9/4)//(nvert*nb*nb*16))
    out = np.zeros((len(energies),len(comps)),dtype=np.complex128)
    t["elements"] = 0.; t["kernel"] = 0.
    for c0 in range(0,len(comps),cblock):
        sl = slice(c0,c0+cblock)
        t0 = time.perf_counter()
        G = elements(U1,U2,P,ia[sl],ib[sl])
        G = G.reshape(nkk,nb,nb,-1)
        sc._multiplet_average_pairs(E1,E2,G,tol)
        if nf>0:
            Gv = elements(V1f,V2f,P,ia[sl],ib[sl])
            G = np.concatenate([G,Gv])
        t["elements"] += time.perf_counter()-t0
        t0 = time.perf_counter()
        out[:,sl] = response_sources_sim(simall,frac,E1all,E2all,G,
                np.asarray(energies,dtype=float),delta,T)
        t["kernel"] += time.perf_counter()-t0
        del G
    if info is not None:
        info.update({"times":t,"flagged":nf,"simplices":ns,
                "side_k":s1,"side_kq":s2,"blocks":-(-len(comps)//cblock)})
    return out.reshape(len(energies),ni,ni)


def rel(a,b):
    return float(np.max(np.abs(a-b))/np.max(np.abs(b)))


# ------------------------------------------------------------ tasks

def task_check():
    """the plumbing: rotation=0 against the library at a generic q, where
    nothing is degenerate except the spin partners, and the library
    against itself at a mesh q"""
    out = []
    for spin in (False,True):
        h = supercell(spin=spin,random=0.2); P = projectors(h)
        for q,nk in (([0.137,0.211],8),([0.137,0.211],12)):
            for (delta,T) in ((0.,0.),(1e-2,1e-2)):
                en = [0.,0.3]
                a = chi_library(h,q+[0.],nk,en,delta,T,P)
                b = chi_proto(h,q,nk,en,delta,T,P,rotation=0)
                c = chi_proto(h,q,nk,en,delta,T,P,rotation=2)
                out.append(dict(spin=spin,q=q,nk=nk,delta=delta,T=T,
                        off_vs_library=rel(b,a),kp_vs_library=rel(c,a)))
                print(out[-1],flush=True)
    return out


def task_direction(spin=False,nks=(4,8,16,32),eps=1e-6):
    """the response at q0 + eps u for eight directions u and at q0, in
    units of the largest entry"""
    h = supercell(spin=spin); P = projectors(h)
    q0 = np.array([0.25,0.5])
    th = 2*np.pi*np.arange(8)/8+0.3
    us = [np.array([0.6,0.8]),np.array([-0.6,-0.8])]+[np.array([np.cos(t),
            np.sin(t)]) for t in th]
    res = []
    for nk in nks:
        for (delta,T) in ((0.,0.),(1e-2,1e-2)):
            en = [0.,0.3]
            row = dict(spin=spin,nk=nk,delta=delta,T=T)
            for name,f in (("library",lambda q: chi_library(h,list(q)+[0.],nk,
                    en,delta,T,P)),
                    ("polar",lambda q: chi_proto(h,q,nk,en,delta,T,P,1)),
                    ("kp",lambda q: chi_proto(h,q,nk,en,delta,T,P,2))):
                c0 = f(q0)
                cs = [f(q0+eps*u) for u in us]
                scale = np.max(np.abs(c0))
                spread = max(np.max(np.abs(a-b)) for a in cs for b in cs)/scale
                pm = np.max(np.abs(cs[0]-cs[1]))/scale
                dev = max(np.max(np.abs(a-c0)) for a in cs)/scale
                mean = np.max(np.abs(np.mean(cs,axis=0)-c0))/scale
                row[name] = dict(u_minus_u=pm,spread8=spread,max_dev_q0=dev,
                        mean_minus_q0=mean,largest=float(scale))
            # the values at q0 of the three, against each other
            row["q0_kp_vs_library"] = rel(chi_proto(h,q0,nk,en,delta,T,P,2),
                    chi_library(h,[0.25,0.5,0.],nk,en,delta,T,P))
            print(json.dumps(row),flush=True)
            res.append(row)
    return res


def task_reference(model,q,nks,energies,delta,T):
    h = models[model](); out = {}
    for nk in nks:
        t0 = time.perf_counter()
        c = chi_grid(h,q+[0.],nk,energies,delta,T)
        out[nk] = c
        print("grid",model,q,nk,time.perf_counter()-t0,flush=True)
        np.save("ref_%s_%s_%d.npy"%(model,"_".join(str(x) for x in q),nk),c)
    return out


models = {"sc":lambda: supercell(),
          "scspin":lambda: supercell(spin=True),
          "scmod":lambda: supercell(modulation=0.1),
          "scrand":lambda: supercell(random=0.2)}


def task_convergence(model,q,nks,energies,delta,T,ref):
    h = models[model](); P = projectors(h)
    R = np.load(ref)
    res = []
    for nk in nks:
        row = dict(model=model,q=q,nk=nk,delta=delta,T=T)
        for name,f in (("library",lambda: chi_library(h,q+[0.],nk,energies,
                delta,T,P)),
                ("kp",lambda: chi_proto(h,q,nk,energies,delta,T,P,2)),
                ("polar",lambda: chi_proto(h,q,nk,energies,delta,T,P,1))):
            c = f()
            scale = np.max(np.abs(R))
            row[name] = [float(np.max(np.abs(c[i]-R[i]))/scale)
                    for i in range(len(energies))]
            np.save("conv_%s_%s_%s_%d_%g.npy"%(model,name,"_".join(str(x)
                    for x in q),nk,T),c)
        print(json.dumps(row),flush=True)
        res.append(row)
    return res


def task_static(model,q,nks):
    """the static response at T = delta = 0, the library and the kp
    prototype, saved for an extrapolation in 1/nk^2"""
    h = models[model](); P = projectors(h)
    for nk in nks:
        a = chi_library(h,q+[0.],nk,[0.],0.,0.,P)
        b = chi_proto(h,q,nk,[0.],0.,0.,P,2)
        np.save("static_%s_%s_%d.npy"%(model,"_".join(str(x) for x in q),nk),
                np.array([a,b]))
        print(nk,rel(b,a),flush=True)


def task_cost(model,q,nk,energies,delta,T,passes=2):
    h = models[model](); P = projectors(h)
    res = []
    for ip in range(passes):
        for name in ("library","kp"):
            info = {}
            t0 = time.perf_counter()
            if name=="library": chi_library(h,q+[0.],nk,energies,delta,T,P)
            else: chi_proto(h,q,nk,energies,delta,T,P,2,info=info)
            dt = time.perf_counter()-t0
            row = dict(model=model,q=q,nk=nk,nw=len(energies),delta=delta,
                    T=T,code=name,wall=dt,pass_=ip,info=info,
                    load=open("/proc/loadavg").read().split()[0])
            print(json.dumps(row),flush=True)
            res.append(row)
    return res


if __name__=="__main__":
    task = sys.argv[1]
    if task=="check": task_check()
    elif task=="direction":
        task_direction(spin=(sys.argv[2]=="spin"),
                nks=tuple(int(x) for x in sys.argv[3].split(",")),
                eps=float(sys.argv[4]) if len(sys.argv)>4 else 1e-6)
    elif task=="reference":
        task_reference(sys.argv[2],[float(x) for x in sys.argv[3].split(",")],
                [int(x) for x in sys.argv[4].split(",")],[0.,0.5,1.5],
                1e-2,1e-2)
    elif task=="convergence":
        task_convergence(sys.argv[2],[float(x) for x in sys.argv[3].split(",")],
                [int(x) for x in sys.argv[4].split(",")],[0.,0.5,1.5],
                1e-2,1e-2,sys.argv[5])
    elif task=="static":
        task_static(sys.argv[2],[float(x) for x in sys.argv[3].split(",")],
                [int(x) for x in sys.argv[4].split(",")])
    elif task=="cost":
        T = float(sys.argv[5])
        en = [0.] if T==0. else [0.,0.5,1.5]
        task_cost(sys.argv[2],[float(x) for x in sys.argv[3].split(",")],
                int(sys.argv[4]),en,T,T)
