

def topological_heavy_fermion_1d(te=1.0,tl=0.2,tk=0.3):
    """Return the Hamiltonian of the model
    of a one-dimensional topological
    heavy-fermion"""
    from .. import geometry
    ge = geometry.square_ribbon(2) # square ribbon
    ge = ge.get_supercell(2) ; ge.center()
    
    def tij(ri,rj):
        r0 = (ri+rj)/2.
        dr = ri-rj
        dr2 = dr.dot(dr)
        if 0.9<dr2<1.1:
            if r0[1]<-0.2:
                if r0[0]<0.2: return te
                else: return te # hopping in the
            if r0[1]>0.2:
                if r0[0]<0.2: return tl
                else: return tl # hopping in the
        if 1.9<dr2<2.1: # hopping between localized and extended modes
            if ri[1]*rj[1]<0.:
                if r0[0]>0.: return tk
                else: return -tk
        return 0.
    he = ge.get_hamiltonian(tij=tij,has_spin=False)
    return he # return Hamiltonian



