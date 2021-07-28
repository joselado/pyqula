# routines to compute magnetic exchange
import numpy as np




def NN_exchange(h,J=0.1,nk=2,num_bands=None,full_energy=False,
        mode="supercell",filling=None):
    """Compute the nearest neighbor exchange using
    a brute force algorithm"""
    if not full_energy:  nbands = num_bands # just the same method for energy
    else: nbands = None # enforce full calculation
    from . import ldos
    d = ldos.ldos_density(h,nk=nk,delta=1e-3,num_bands=num_bands) # density
    if mode=="supercell":
        hs = h.supercell([2,1,1]) # get a supercell
        ds0 = J*np.concatenate([d,d]) # FE configuration
        ds1 = J*np.concatenate([d,-d]) # AF configuration
        if h.has_spin: # spinful version
            ds0 = np.array([0*ds0,0*ds0,ds0]).T # magnetizations
            ds1 = np.array([0*ds1,0*ds1,ds1]).T # magnetizations
            hs0 = hs.copy()
            hs1 = hs.copy()
            hs0.add_zeeman(ds0) # add the Zeeman field
            hs1.add_zeeman(ds1) # add the Zeeman field
            e0 = hs0.get_total_energy(nk=nk,nbands=nbands) # GS energy 
            e1 = hs1.get_total_energy(nk=nk,nbands=nbands) # GS energy 
            return (e1-e0)/2.
        else: # spinless version
            hs0u = hs.copy()
            hs0d = hs.copy()
            hs1u = hs.copy()
            hs1d = hs.copy()
            hs0u.add_onsite(ds0)
            hs0d.add_onsite(-ds0)
            hs1u.add_onsite(ds1)
            hs1d.add_onsite(-ds1)
            gete = lambda h0: h0.get_total_energy(nk=nk,nbands=nbands) 
            return (gete(hs1u) + gete(hs1d) - gete(hs0u) - gete(hs0d))/2.
    elif mode=="spiral": # spin spiral mode
        dm = J*np.array([d,0*d,0*d]).T # magnetizations
        h = h.copy() # copy Hamiltonian
        h.turn_spinful() # add spin
        h.add_zeeman(dm) # add exchange field
        if filling is not None: h.set_filling(filling,nk=nk) # set filling
        h0 = h.copy() # get the ferro
        h1 = h.copy() # get the AF
        h1.generate_spin_spiral(vector=[0,0,1],qspiral=[1,0,0])
        e0 = h0.get_total_energy(nk=nk,nbands=nbands) # GS energy 
        e1 = h1.get_total_energy(nk=nk,nbands=nbands) # GS energy 
        return e1-e0
    else: raise # unrecognized mode



