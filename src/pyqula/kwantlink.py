from __future__ import print_function
import kwant



def transport(ht,energy):
  """Define a Kwant heterostructure using an input one"""
  # create kwant objects
  lat = kwant.lattice.square()
  sys = kwant.Builder() 
  # create the two leads
  sym_L = kwant.TranslationalSymmetry((-1, 0))
  sym_R = kwant.TranslationalSymmetry((1, 0))
  lead_L = kwant.Builder(sym_L) # create the lead
  lead_R = kwant.Builder(sym_R) # create the lead
  lead_R[lat(0,0)] = ht.right_intra
  lead_L[lat(0,0)] = ht.left_intra
  lead_R[lat(0,0),lat(1,0)] = ht.right_inter
#  lead_R[lat.neighbors()] = ht.right_inter
  lead_L[lat(0,0),lat(-1,0)] = ht.left_inter
#  lead_L[lat.neighbors()] = ht.left_inter
  # The scattering region will be three sites
  # add the onsites of the leads
  if not ht.block_diagonal: # dense scattering region
    # and create the coupling to the central region
    # create central and scattering regions
    sys[lat(-1,0)] = ht.left_intra
    sys[lat(1,0)] = ht.right_intra
    sys[lat(0,0)] = ht.central_intra
    sys[lat(0,0), lat(-1,0)] = ht.left_coupling
    sys[lat(0,0), lat(1,0)] = ht.right_coupling
    # now attach the two leads
    sys.attach_lead(lead_L, lat(-1, 0))
    sys.attach_lead(lead_R, lat(1, 0))
  else: # block diagonal form
    nb = len(ht.central_intra)
    sys[lat(-1,0)] = ht.left_intra
    sys[lat(nb,0)] = ht.right_intra
    for i in range(nb):
      sys[lat(i,0)] = ht.central_intra[i][i]
    for i in range(nb-1):
      sys[lat(i,0),lat(i+1,0)] = ht.central_intra[i][i+1]
    sys[lat(0,0), lat(-1,0)] = ht.left_coupling
    sys[lat(nb-1,0), lat(nb,0)] = ht.right_coupling
    sys.attach_lead(lead_L, lat(-1, 0))
    sys.attach_lead(lead_R, lat(nb, 0))
  sys = sys.finalized()
#  kwant.plot(sys)
  smatrix = kwant.smatrix(sys, energy)
  return smatrix.transmission(1, 0)

