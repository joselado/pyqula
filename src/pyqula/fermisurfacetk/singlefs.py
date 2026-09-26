import numpy as np
from .. import algebra
from .. import parallel
import scipy.sparse.linalg as slg

arpack_tol = 1e-5
arpack_maxiter = 10000

def fermi_surface(h,write=True,output_file="FERMI_MAP.OUT",
                    e=0.0,nk=50,nsuper=1,reciprocal=True,
                    k0 = np.array([0.,0.]),
                    delta=None,refine_delta=1.0,operator=None,
                    mode='eigen',num_waves=10,info=False,
                    primal_mesh=False,
                    backend="grid",tolerance=1e-3,**qtci_kwargs):
    """Calculates the Fermi surface of a 2d system.

    backend: "grid" (default) evaluates the spectral weight at every point
    of the nk x nk mesh. "qtci" instead uses qutecipy's quantics tensor
    cross interpolation (QTCI): kx and ky are each encoded as R=ceil(log2(nk))
    bits, so the BZ mesh becomes a tensor with 2*R small (dimension-2) legs
    instead of 2 big (dimension-nk) legs. A plain 2-leg TCI on the raw mesh
    gains nothing (a full pivot search over a 2-leg tensor already has to
    scan the whole matrix), but pivot search over many small legs is cheap,
    so this quantics encoding is what actually lets TCI skip most of the
    nk x nk diagonalizations -- worthwhile once nk is large and the Fermi
    surface is reasonably smooth/compressible; a very jagged, near-full-rank
    surface will gain little and just pay the TCI overhead. The quantics
    mesh is internally rounded up to the nearest power of two (2**R); if
    nk isn't already one, the reconstructed mesh is interpolated back down
    to the exact nk x nk grid the caller asked for, so the returned
    (kx,ky,kdos) always matches the "grid" backend's shape/coordinates.

    primal_mesh: for a supercell built with store_primal=True, draw the
    mesh (and read k0) in the Brillouin zone of the primal cell, and
    compute every point at its image in the supercell's reduced
    coordinates, k_S = M@k_0 (see unfolding.get_primal_mesh_map). With
    operator="unfold" this is the unfolded Fermi surface on the same mesh,
    in the same coordinates, as the primal cell's own, for any supercell
    matrix M."""
    operator = h.get_operator(operator) # get the operator
    if operator is not None: # operator given
        if not operator.linear:
            if mode=="full": mode = "eigen"
    else: # no operator given
        if mode=="full":
            operator = np.array(np.identity(h.intra.shape[0]))
    if h.dimensionality!=2: # continue if two dimensional
        raise ValueError("the Fermi surface is only defined for 2d "
                "Hamiltonians")
    hk_gen = h.get_hk_gen() # gets the function to generate h(k)
    from ..klist import int2dims
    nsupers = int2dims(nsuper) # get the array
    nks = int2dims(nk) # get the array
    kxs = np.linspace(-nsupers[0],nsupers[0],nks[0])  # generate kx
    kys = np.linspace(-nsupers[1],nsupers[1],nks[1])  # generate ky
    iden = np.identity(h.intra.shape[0],dtype=np.complex128)
    kxout = []
    kyout = []
    if reciprocal: R = h.geometry.get_k2K_generator() # get function
    else:  R = lambda x: x
    if primal_mesh: # mesh and k0 in the primal cell's Brillouin zone
        from ..unfolding import get_primal_mesh_map
        R = get_primal_mesh_map(h.geometry,reciprocal=reciprocal,k0=k0)
        k0 = np.zeros(3) # the shift is already in R
    # setup a reasonable value for delta
    if delta is None:  delta = 3./refine_delta*2./nk
    #### function to calculate the weight ###
    # Every mode returns sum_n <n|A|n> delta/((e-E_n)^2+delta^2), the
    # weight that mode='eigen' without an operator has always returned (pi
    # times the spectral function, without the 1/pi of a DOS). The operator
    # branches of 'eigen' and 'lowest' go through h.get_dos, which does
    # apply the 1/pi, and 'full' takes G^R-G^A, which is twice G^R; so they
    # used to come out 1/pi and 2 times that, and the same unfolded or
    # spin-resolved Fermi surface changed scale by 2*pi between two modes
    if mode=='full': # use full inversion
      def get_weight(hk,k=None):
        gf = algebra.inv((e+1j*delta)*iden - hk) # get green function
        gf = gf - algebra.inv((e-1j*delta)*iden - hk) # get green function
        if callable(operator): # callable operator
           tdos = -(operator(gf,k=k)).imag # get imaginary part
        else: tdos = -(operator@gf).imag # get imaginary part
        return np.trace(tdos).real/2. # G^R-G^A counts every pole twice
    elif mode=='eigen': # use full diagonalization
      def get_weight(hk,k=None):
        if operator is None:
            es = algebra.eigvalsh(hk)
            return np.sum(delta/((e-es)**2+delta**2)) # return weight
        else: # using an operator
            tmp,ds = h.get_dos(ks=[k],operator=operator,
                        write=False,
                        energies=[e],delta=delta)
            return np.pi*ds[0] # undo get_dos's 1/pi, see above
    elif mode=='lowest': # use sparse diagonalization
      def get_weight(hk,k=None,**kwargs):
        if operator is None: # without operator
            if num_waves>=hk.shape[0]-1: # more than ARPACK can return,
                # fall back to all the states, as get_bands does for the
                # operator branch below, rather than raise inside scipy
                es = algebra.eigvalsh(hk)
                return np.sum(delta/((e-es)**2+delta**2)) # return weight
            es,waves = slg.eigsh(hk,k=num_waves,sigma=e,
                    tol=arpack_tol,which="LM",
                              maxiter = arpack_maxiter)
            return np.sum(delta/((e-es)**2+delta**2)) # return weight
        else: # using an operator
            htmp = h.copy() # make a dummy copy
            htmp.shift_fermi(-e) # shift the Fermi energy
            tmp,ds = htmp.get_dos(ks=[k],operator=operator,
                        write=False,
                        energies=[0.],delta=delta,
                        num_bands=num_waves)
            return np.pi*ds[0] # undo get_dos's 1/pi, see above
    elif mode=='det': # use determinant method, this is not too stable
        if operator is not None:
            raise NotImplementedError("the determinant mode of the Fermi "
                    "surface does not support an operator")
        else: # None operator
            iden = algebra.identity(h.intra)
            def get_weight(hk,k=None,**kwargs):
                hk0 = hk - e*iden # shift by the energy
                return 1./(np.abs(algebra.det(hk0))+delta)

    else: # unrecognized mode
        raise ValueError("unknown mode; the Fermi surface accepts 'full', "
                "'eigen', 'lowest' and 'det'")
  
  ##############################################
  
  
    # setup the operator
    k0 = np.array([k0[0],k0[1],0.])
    if backend=="qtci": # quantics-encoded adaptive sampling with qutecipy
        from ..qutecipytk import crossinterpolate2
        from ..qutecipytk.tensortrain.core import tensortrain
        from ..qutecipytk.tensortrain.cachedfunction import CachedFunction
        from ..qutecipytk.quantics.discretized import DiscretizedGrid, quantics_function
        nbits = max(1,int(np.ceil(np.log2(max(len(kxs),len(kys)))))) # bits/axis, rounds nk up to 2**nbits
        qgrid = DiscretizedGrid.from_resolutions(["x","y"],[nbits,nbits],
                lower_bound=[kxs[0],kys[0]],upper_bound=[kxs[-1],kys[-1]],
                unfoldingscheme="interleaved",includeendpoint=True)
        def getf_qtci(x,y): # function of the (continuous) mesh coordinates
            r = np.array([x,y,0.])
            k = R(r) + k0
            hk = hk_gen(k) # get hamiltonian
            return get_weight(hk,k=k)
        qf = quantics_function(np.float64,qgrid,getf_qtci)
        localdims = qgrid.localdimensions() # 2*nbits legs, each of dimension 2
        # pivot search revisits the same indices many times over; cache so
        # each quantics index is only diagonalized once
        qf = CachedFunction(np.float64,qf,localdims)
        tci,ranks,errors = crossinterpolate2(np.float64,qf,localdims,
                tolerance=tolerance,**qtci_kwargs)
        full = tensortrain(tci).fulltensor() # reconstruct the full mesh, no extra evaluations
        # de-interleave the alternating (xbit0,ybit0,xbit1,ybit1,...) legs
        # back into (all x bits, all y bits), each block MSB-first, so a
        # plain C-order reshape recovers the binary grid index per axis
        xaxes = list(range(0,2*nbits,2))
        yaxes = list(range(1,2*nbits,2))
        full = np.transpose(full,xaxes+yaxes)
        n = 2**nbits
        kdos_q = full.reshape(n,n)
        kxsq = np.array(qgrid.grid_origcoords(0))
        kysq = np.array(qgrid.grid_origcoords(1))
        if n==len(kxs) and n==len(kys) and np.allclose(kxsq,kxs) and np.allclose(kysq,kys):
            # nk was already a power of two: the quantics mesh *is* the requested mesh
            kxout = np.repeat(kxsq,n) # x is the slower (outer) index, matching the reshape's C order
            kyout = np.tile(kysq,n)
            kdos = kdos_q.flatten()
        else:
            # nk was rounded up to the nearest power of two internally; bring
            # the reconstructed mesh back down to the resolution the caller
            # asked for by interpolating on the (finer) quantics mesh
            from scipy.interpolate import RegularGridInterpolator
            interp = RegularGridInterpolator((kxsq,kysq),kdos_q,bounds_error=False,fill_value=None)
            KX,KY = np.meshgrid(kxs,kys,indexing="ij") # x outer, y inner, matching the grid backend
            kxout = KX.flatten()
            kyout = KY.flatten()
            kdos = interp((KX,KY)).flatten()
    else: # brute-force evaluation on every point of the mesh
        rs = [] # empty list
        for x in kxs:
          for y in kys:
            rs.append([x,y,0.]) # store
        def getf(r): # function to compute FS
            k = R(r) + k0
            hk = hk_gen(k) # get hamiltonian
            return get_weight(hk,k=k)
        rs = np.array(rs) # transform into array
        kxout = rs[:,0] # x coordinate
        kyout = rs[:,1] # y coordinate
        if mode=='eigen' and operator is None:
            # batched, numba-parallel path -- no interprocess dispatch
            from ..htk.eigenvectors import peigvalsh_bloch
            ks = np.array([R(r)+k0 for r in rs]) # kpoints, change of basis applied
            kdos = np.zeros(len(rs),dtype=np.float64)
            batch_size = 64
            for i0 in range(0,len(rs),batch_size): # loop over batches of kpoints
                kbatch = ks[i0:i0+batch_size]
                es_batch = peigvalsh_bloch(hk_gen,kbatch) # diagonalize the whole batch in parallel
                for ii in range(len(kbatch)):
                    es = es_batch[ii]
                    kdos[i0+ii] = np.sum(delta/((e-es)**2+delta**2))
        elif parallel.cores==1: # serial execution
            kdos = np.zeros(len(rs),dtype=np.float64) # empty array
            for ir in range(len(rs)): # loop
                kdos[ir] = getf(rs[ir]) # store
        else: # parallel execution
            kdos = parallel.pcall(getf,rs) # compute all
    if write:  # optionally, write in file
      f = open(output_file,"w")
      for (x,y,d) in zip(kxout,kyout,kdos):
        f.write(str(x)+ "   "+str(y)+"   "+str(d)+"\n")
      f.close() # close the file
    return (kxout,kyout,np.array(kdos)) # return result


