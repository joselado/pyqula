import numpy as np
from ..htk import fusion


def H2HFH(h,JK=0.0,J=0.):
    """Given a certain geometry, generate a new geometry with heavy femrion sites"""
    if h.has_eh:
        print("Not implemented with superconductivity")
        raise
    if h.has_kondo:
        print("This Hamiltonian already has Kondo sites")
        raise
    width = np.max(h.geometry.r[:,2]) - np.max(h.geometry.r[:,2])
    if width>1e-4: 
        print("Not implemented for non-2D Hamiltonians")
        raise # not implemented for not 2D 
    h.turn_spinful() ; h.turn_dense()
    g = h.geometry # get the geometry
    gl = g.copy() ; gl.r[:,2] += 0.5 ; gl.r2xyz()
    hl = gl.get_hamiltonian(has_spin=True,tij = [J])
    hl.set_filling(0.5) # set to half filling
    # Hamiltonian of the extended modes
    gd = g.copy() ; gd.r[:,2] -= 0.5 ; gd.r2xyz()
    h.geometry = gd
    hd = h.copy() 
    # now the interlayer coupling
    gt = gd + gl
    def ft(r1,r2):
        if r1[2]*r2[2]<0.:
            dr = r1-r2
            dr2 = dr.dot(dr)
            if 0.9<dr2<1.1: return JK
        return 0.
    ht = gt.get_hamiltonian(has_spin=True,tij=ft)
    h = fusion.hamiltonian_fusion(hd,hl) + ht
    h.has_kondo = True # now with Kondo sites
    return h


def get_operator(self,name,**kwargs):
    """Return operators for systems with Kondo sites"""
    if name=="kondo_sites":
        return self.get_operator(lambda r: 1.*(r[2]>0.))
    elif name=="dispersive_electrons":
        return self.get_operator(lambda r: 1.*(r[2]<0.))
    else: raise



def add_onsite(self,ons):
    if not self.has_kondo: raise
    m = self.copy() # copy Hamiltonian
    dis = m.get_operator("dispersive_electrons") # operator for dispersive
    intra = m.add_onsite(ons).intra@dis # add onsite only to disp electrons
    self.intra = self.intra + intra # include contribution


