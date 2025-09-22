# generates diffeent types of klist

import numpy as np
import scipy.linalg as lg
from . import geometry
from numba import jit

def get_klist(g,ns,nk=100):
    """Return a klist from a list of names"""
    # generate dictionary
    kdict = dict()
    kdict["G"] = [0.,0.,0.]
    if np.abs(g.a1.dot(g.a2))<0.001: # square lattice
      kdict["X"] = [0.5,0.,0.]
      kdict["Y"] = [0.,0.5,0.]
      kdict["M"] = [0.5,0.5,0.]
    else: # triangular lattice
      angle = py_ang(g.a1,g.a2) # angle between vectors
      kdict["M1"] = [1./2.,1./2.,0.]
      kdict["M2"] = [0.,1./2.,0.]
      kdict["M3"] = [1./2.,0.,0.]
      if np.abs(angle - np.pi*2./3.)<0.001: # first type
        kdict["K"] = [1./3.,1./3.,0.]
        kdict["K'"] = [2./3.,2./3.,0.]
        kdict["M"] = [1./2.,0.,0.]
      else: # second type
        kdict["K"] = [1./3.,-1./3.,0.]
        kdict["K'"] = [2./3.,-2./3.,0.]
        kdict["M"] = [1./2.,0.,0.]
    # create the different vertex
    kv = np.array([kdict[n] for n in ns]) # get the different ones
    # create the different k-points
    ks = [] # empty list
    kinds = [0] # initial one
    for i in range(len(ns)-1): # loop over vertex
        kn = kv[i+1]-kv[i] # vector linking the two
        k0 = kv[i] # first vector
        knorm = np.sqrt(kn.dot(kn)) # norm
        nki = int(nk*knorm) # number of points
        kinds.append(nki+sum(kinds)) # store
        dk = np.linspace(0.,1.,nki,endpoint=False) # create increment
        for ik in dk:
            ks.append(k0 + ik*kn) # new vector
    ks.append(kv[len(ns)-1]) # last one
    fbl = open("BANDLINES.OUT","w")
    for i in range(len(ns)): 
        fbl.write(str(kinds[i])+" "+ns[i]+"\n")
    return ks # return vector





def py_ang(v1, v2):
  """ Returns the angle in radians between vectors 'v1' and 'v2'    """
  v1,v2 = v1.real,v2.real
  cosang = np.dot(v1, v2)
  sinang = lg.norm(np.cross(v1, v2))
  return np.arctan2(sinang, cosang)







def default_kpath(g,nk=400):
  """ Input is geometry"""
  if g.dimensionality==0: return [0.] # return gamma point
  elif g.dimensionality==1: 
    return [np.array([k,0,0]) for k in np.linspace(0,1,nk)] # normal path
  elif g.dimensionality > 1:
    b1 = np.array([1.,0.,0.])
    b2 = np.array([0.,1.,0.])
    (G1,G2) = geometry.get_reciprocal2d(g.a1,g.a2)
    bm = b1 + b2 # wave-vector along M
    angle = py_ang(g.a1,g.a2) # angle between vectors
    if np.abs(angle - np.pi*2./3.)<0.001:  
      bm = np.array([1.,1.,0.]) # temporal fix
#    else np.abs(angle - np.pi*2./6.):  
    else:
      bm = np.array([1.,-1.,0.]) # temporal fix
#    print("Path along",bm)
#    b2 = np.array([.5,np.sqrt(3)/2])
#    b2 = np.array([0.,-1.])
    fk = open("KPOINTS_BANDS.OUT","w")  
#    fk.write(str(nk)+"\n") # number of kpoints
    k = np.array([0.,0.,0.]) # old kpoint
    kout= []
    for i in range(nk):
      k += bm /(nk) # move kpoint 
      fk.write(str(k[0])+"   "+str(k[1])+"\n    ")
      kout.append(k.copy()) # store in array
    fk.close()
    # write bandlines
    fbl = open("BANDLINES.OUT","w")
    if np.abs(g.a1.dot(g.a2))<0.001: # square lattice
      fbl.write("0   \Gamma\n")
      fbl.write(str(nk/2)+"   M\n")
      fbl.write(str(nk)+"   \Gamma\n")

    else: # triangular lattice
      fbl.write("0   \Gamma\n")
      fbl.write(str(nk/3)+"   K\n")
      fbl.write(str(nk/2)+"   M\n")
      fbl.write(str(2*nk/3)+"   K'\n")
      fbl.write(str(nk)+"   \Gamma\n")
    fbl.close()
    return kout


default = default_kpath # for compatibility

# def full_bz(g)


# def gmxg(nk=100):



def gen_default(k):
  """ Return a function which generates the path"""
  b1 = np.array([1.,0.])
  b2 = np.array([0.,1.])
  return k*(b1+b2) # return kpoint 


def path_GKMKG(g,nk):
  """Generate a path G-K-M-K'-G"""
  a1 = g.a1 # vector
  a2 = g.a2 # vector
  raise



def write_klist(kl,output_file="klist.in"):
  """ Writes a set of 2D kpoints in a file"""
  fk = open(output_file,"w")  
  fk.write(str(len(kl))+"\n") # number of kpoints
  for k in kl:
    fk.write(str(k[0])+"   "+str(k[1])+"\n    ")
  fk.close()


def custom_klist(kp = None,nk=100,write=True,scale=True):
  if kp==None:
    kp = [[0.,0.,0.],[.5,.0,0.],[.5,.5,0.],[.0,.0,0.]]  # define points
  kp = [np.array(k) for k in kp] # convert to arrays
  ks = [] # empty list of kpoints
  for i in range(len(kp)-1): # loop over pairs
    dk = kp[i+1] - kp[i]
    kk = np.sqrt(dk.dot(dk)) # norm of the vector
    if scale:
      steps = np.linspace(0.,1.,int(nk*kk),endpoint=False) # number of points
    else:
      steps = np.linspace(0.,1.,nk,endpoint=False) # number of points
    for s in steps:
      ks += [kp[i] + dk*s] # add kpoint
  if write:
    write_klist(ks)
  return ks

custom = custom_klist # same function


def kx(g,nk=400):
  """ Input is geometry"""
  if g.dimensionality == 2:
    b1 = np.array([1.,0.])
    b2 = np.array([0.,1.])
    fk = open("klist.in","w")  
    fk.write(str(nk)+"\n") # number of kpoints
    k = -b1/2 # old kpoint
    kout= []
    for i in range(nk):
      k += (b1) /(nk) # move kpoint 
      fk.write(str(k[0])+"   "+str(k[1])+"\n    ")
      kout.append(k.copy()) # store in array
    fk.close()
    # write bandlines
    fbl = open("BANDLINES.OUT","w")
    fbl.write("0   X_1\n")
    fbl.write(str(nk/2)+"   \Gamma\n")
    fbl.write(str(nk)+"   X_1\n")
    fbl.close()
  return kout # return klist



def tr_path(nk=100,d=20,write=True):
  """ Creates the special path to calculate the Z2 invariant"""
  # d is the number of divisions
  w = 1.0/8 # heigh of the path
  ks = [] # initialice list
  dk = 1./nk 
  # full path
  ks += [[il,.5/d] for il in np.arange(-.5-dk,.5/d,dk)] 
  ks += [[.5/d,il] for il in np.arange(.5/d,0-dk,-dk)] 
  ks += [[il,0.-dk] for il in np.arange(.5/d,(d-1)*.5/d,dk)] 
  ks += [[(d-1)*.5/d,il] for il in np.arange(0.-dk,.5/d,dk)] 
  ks += [[il,.5/d] for il in np.arange((d-1)*.5/d,.5+dk,dk)] 
  ks += [[.5+dk,il] for il in np.arange(.5/d,(d-1)*.5/d,dk)] 
  ks2 = [[-k[0],.5-k[1]] for k in ks]
  ks = ks + ks2 # sum the two lists
  if write:
    f = open("klist.in","w")
    f.write(str(len(ks))+"\n") # number of kpoints
    for k in ks:
      f.write(str(k[0]) + "  ")
      f.write(str(k[1]) + "\n")
    f.close()



def tr_klist(nk=100,d=20):
  """ Creates the special path to calculate the Z2 invariant"""
  # d is the number of divisions
  w = 1.0/8 # heigh of the path
  lks = [] # initialice list
  dk = 1./nk 
  # full path
  class ks_class: pass
  ks = ks_class() # create object
  lks += [[il,.5/d] for il in np.arange(-.5-dk,.5/d,dk)] 
  lks += [[.5/d,il] for il in np.arange(.5/d,0-dk,-dk)] 
  lks += [[il,0.-dk] for il in np.arange(.5/d,(d-1)*.5/d,dk)] 
  lks += [[(d-1)*.5/d,il] for il in np.arange(0.-dk,.5/d,dk)] 
  lks += [[il,.5/d] for il in np.arange((d-1)*.5/d,.5+dk,dk)] 
  ks.common = [[.5+dk,il] for il in np.arange(.5/d,(d-1)*.5/d,dk)] 
  lks2 = [[-k[0],.5-k[1]] for k in lks]
  ks.path1 = lks  # first path
  ks.path2 = lks2  # second path
  return ks




def default_v2(g,nk=400):
  """ Input is geometry"""
  if g.dimensionality == 2:
    b1 = np.array([1.,0.])
    b2 = np.array([0.,1.])
    fk = open("klist.in","w")  
    fk.write(str(nk)+"\n") # number of kpoints
    k = np.array([0.,0.]) # old kpoint
    kout= []
    for i in range(nk):
      k += (b1-b2) /(nk) # move kpoint 
      fk.write(str(k[0])+"   "+str(k[1])+"\n    ")
      kout.append(k.copy()) # store in array
    fk.close()
    # write bandlines
    fbl = open("BANDLINES.OUT","w")
    fbl.write("0   \Gamma\n")
    fbl.write(str(nk/3)+"   K\n")
    fbl.write(str(nk/2)+"   M\n")
    fbl.write(str(2*nk/3)+"   K'\n")
    fbl.write(str(nk)+"   \Gamma\n")
    fbl.close()
  else: raise # not implemented




def partial_kmesh(d,nk=10,k0=[0.,0.,0.],f=1.0):
    """Return a partial kmesh, centerer around a certain point"""
    ks = kmesh(d,nk=nk) # initial kmesh
    ks = np.array(ks) # convert to array
    kc = [np.mean(ks[:,i]) for i in range(3)] # center
    k0 = np.array(k0) # convert to array
    ks = [k - kc for k in ks] # center in 0,0,0
    ks = np.array(ks)*f # scale
    ks = [k + k0 for k in ks] # redefine
    return np.array(ks) # return kmesh



def infer_kmesh_dk(ks,d=2):
    den = infer_kmesh_density(ks,d=d)
    return np.sqrt(den)/2.




def infer_kmesh_density(kmesh,d=2):
    """Given a certain kmesh, infer what is the density"""
    ks = np.array(kmesh) # convert to array
    dk = [np.max(ks[:,i]) - np.min(ks[:,i]) for i in range(3)]
    if d==2: # this is a temporal workround
        nk = np.round(np.sqrt(len(ks))) # infer number of points
        dk = np.array(dk)*nk/(nk-1)
        den = dk[0]*dk[1]/(nk*nk)
    return den









def kmesh(dimensionality,nk=10,nsuper=1,
        endpoint=False):
  """Return a mesh of k-points for a certain dimensionality"""
  from .checkclass import number2array
  kp = []
  if nk==1 or dimensionality==0: return [[0.,0.,0.]]
  if dimensionality==1:
    for k1 in np.linspace(0.,nsuper,nk,endpoint=endpoint):
      kp.append([k1,0.,0.]) # store
  elif dimensionality==2:
    nk = number2array(nk) # return an array
    for k1 in np.linspace(0.,nsuper,nk[0],endpoint=endpoint):
      for k2 in np.linspace(0.,nsuper,nk[1],endpoint=endpoint):
        kp.append([k1,k2,0.]) # store
  elif dimensionality==3:
    nk = number2array(nk) # return an array
    for k1 in np.linspace(0.,nsuper,nk[0],endpoint=endpoint):
      for k2 in np.linspace(0.,nsuper,nk[1],endpoint=endpoint):
        for k3 in np.linspace(0.,nsuper,nk[2],endpoint=endpoint):
          kp.append([k1,k2,k3]) # store
  else: raise
  kp = [np.array(k) for k in kp] # to array
  return kp


from .kpointstk.labels import label2k



def get_kpath_labels(g,ks,write=True,**kwargs):
    """Return the k-path"""
    kps = [label2k(g,k) for k in ks] # get the kpoints
    from .kpointstk.locate import k2path
    out = k2path(g,kps,**kwargs) # closest path
    if write: 
        from .kpointstk.bandlines import write_bandlines
        write_bandlines(g,out,ks) # write the lines in a file
    return out


def get_kpath(g,kpath=None,**kwargs):
    """Return a kpath"""
    if kpath is None: 
        return np.array(default(g,**kwargs))
    elif type(kpath[0])==str: 
        return np.array(get_kpath_labels(g,kpath,**kwargs))
    else: return np.array(kpath) # assume is a valid list of vectors


@jit(nopython=True)
def kgrid2d(kxs,kys):
    """Return a list of kvector that create a grid over the inputs"""
    nx = len(kxs)
    ny = len(kys)
    out = np.zeros((nx*ny,3),dtype=float) # create array
    iz = 0 # initialize
    for ix in range(nx):
        x = kxs[ix] # x coordinate
        for iy in range(ny):
            y = kys[iy] # x coordinate
            out[iz,0] = x # store
            out[iz,1] = y # store
            iz += 1 # increase counter
    return out



