import numpy as np
import random


class Alloy():
    """Alloy object, based on geometry"""
    def __init__(self,g):
        """Initialize the object"""
        self.dimensionality = g.dimensionality
        self.r = g.r.copy()
        self.n = len(self.r)
        self.nrep = 1
        self.a1 = g.a1
        self.a2 = g.a2
        self.a3 = g.a3
        self.specie = [0 for ri in self.r] # specie index
        self.nspecie = 1 # number of species
        self.setup_distances()
        self.fenergy = lambda d,i,j: 0.0
        self.setup_interaction()
    def setup_distances(self,**kwargs):
        self.d2 = setup_distances(self,**kwargs) # store square distances
    def setup_interaction(self,f=None):
        """Setup the objects required to compute interactions"""
        if type(f) is dict: f = dict2array(f,ns=self.nspecie)
        if f is None:
            self.get_energy_i = lambda ii: get_energy_i(self,self.fenergy,ii)
        elif callable(f): # input is a function
            self.get_energy_i = lambda ii: get_energy_i(self,f,ii)
        else:
            self.get_energy_i = lambda ii: wrapper_get_energy_i(self,ii,f)
    def get_energy(self):
        return get_energy(self)
    def set_species(self,s):
        self.specie = np.array(s) # specie index
        self.nspecie = len(np.unique(self.specie)) # number of species
    def minimize_energy(self,**kwargs):
        minimize_energy(self,**kwargs)
    def supercell(self,n):
        supercell(self,n)
    def write(self):
        write(self)
    def randomize(self,sp):
        """Randomize species"""
        ss = [0 for i in range(len(self.r))]
        n = len(self.r) # number of sites
        if sum(sp)!=n: raise
        if len(sp)!=2: raise
        ones = random.sample(range(0,n),sp[1])
        for o in ones: ss[o] = 1
        self.set_species(ss)

     


def setup_distances(self,n=1,cut=64):
    """Compute all the possible distances"""
    ds = np.array([0.0])
    r = self.r
    for i in range(-n,n+1):
      for j in range(-n,n+1):
        for k in range(-n,n+1):
            dn = []
            for ri in r:
              for rj in r:
                rk = rj + i*self.a1 + j*self.a2 + k*self.a3
                d = ri-rk
                d = d.dot(d)
#                d = np.array([d])
                dn.append(d)
            ds = np.concatenate([ds,dn]) # store
            ds = np.unique(ds) # retain unique
    ds = np.sort(ds)
    ds = ds[ds<cut]
    return ds # return distances



def get_energy(self,**kwargs):
    """Compute all the possible distances"""
    eout = 0.0
    r = self.r
    for ii in range(len(r)):
      eout += self.get_energy_i(ii,**kwargs)
    return eout 

def get_energy_i(self,f,ii,**kwargs):
    eout = 0.0
    r = self.r
    print(len(r)); exit()
    for jj in range(len(r)):
        eout += get_energy_ij(self,f,ii,jj,**kwargs)
    return eout



def get_energy_ij(self,f,ii,jj,n=1):
    etot = 0.0
    s = self.specie
    r = self.r
    ri = r[ii]
    rj = r[jj]
    si = s[ii]
    sj = s[jj]
    for i in range(-n,n+1):
      for j in range(-n,n+1):
        for k in range(-n,n+1):
                rk = rj + i*self.a1 + j*self.a2 + k*self.a3
                d = ri-rk
                d = d.dot(d)
                etot += f(d,si,sj) # function giving the energy
    return etot # return distances

def different_random_index(self):
    while True:
       ii = np.random.randint(0,self.n)
       jj = np.random.randint(0,self.n)
       if self.specie[ii]!=self.specie[jj]: return (ii,jj)


def minimize_energy(self,n=1000,mode="random",**kwargs):
    """Minimize the energy"""
    if mode=="random":
      eold = self.get_energy()
      for i in range(n):
          (ii,jj) = different_random_index(self)
          eold = single_update(self,eold,ii,jj,**kwargs)
    elif mode=="brute":
        def itene():
          eold = self.get_energy()
          for i in range(self.n):
            for j in range(self.n):
                if self.specie[i]==self.specie[j]: continue # next one
                eold = single_update(self,eold,i,j,**kwargs) # compute
          return eold
        eold = self.get_energy()
        while True:
            enew = itene() # iterate
            if abs(eold-enew)<1e-2: break
            eold = enew
    else: raise # not recognized

def single_update(self,eold,ii,jj,T=1e-7):
        s = self.specie.copy()
        sold = self.specie.copy()
        s[ii] = sold[jj]
        s[jj] = sold[ii]
        eijold = self.get_energy_i(ii) + self.get_energy_i(jj)
        self.specie = s # rewrite
        eijnew = self.get_energy_i(ii) + self.get_energy_i(jj)
        de = (eijnew - eijold)*2
        e = eold + de
        p = np.random.random()
        if de<0.0 or np.exp(-de/T)>p: # accept
            eold = e
            sold = s
            self.specie = s
            print("New energy ",e)
        else: self.specie = sold # next try
        return eold


from numba import jit

@jit(nopython=True)
def fn_get_energy_i(r,s,n,a1,a2,a3,ii,em,dis):
    """Specialized function for first neighbor interaction only"""
    etot = 0.0
    nn = len(dis) # number of neighbors
    dmax = np.max(dis) # maximum distance
    ri = r[ii]
    si = int(s[ii])
    for jj in range(len(r)): # loop over positions
      rj = r[jj]
      sj = int(s[jj])
      for i in range(-n,n+1):
        for j in range(-n,n+1):
          for k in range(-n,n+1):
                  rk = rj + i*a1 + j*a2 + k*a3
                  d = ri-rk
                  dd = np.sum(d*d)
                  if (dd-dmax)>1e-4: continue # too far
                  for nni in range(nn): # loop
                      if abs(dd-dis[nni])<1e-4: # this neighbor
                        etot = etot+ em[nni][si][sj] 
    return etot # return distances


def wrapper_get_energy_i(self,ii,f):
  f = np.array(f) # to array
  if len(f.shape)==2: f = np.array([f]) # redefine
  nn = len(f) # number of neighbors
  return fn_get_energy_i(self.r,self.specie,
                    self.nrep,self.a1,self.a2,self.a3,ii,f,
                    self.d2[1:nn+1])




def supercell(self,n):
    ro = []
    so = []
    for i in range(n):
      for j in range(n):
        for k in range(n):
            for ii in range(len(self.r)):
                ri = self.r[ii] +i*self.a1 + j*self.a2 + k*self.a3
                ro.append(ri)
                so.append(self.specie[ii])
    self.r = np.array(ro)
    self.a1 = n*self.a1
    self.a2 = n*self.a2
    self.a3 = n*self.a3
    self.n = len(self.r)
    self.specie = np.array(so)
    self.nspecie = len(self.specie)
    self.setup_distances()




def write(self,scale=1.0):
    r = self.r/scale
    np.savetxt("ALLOY.OUT",np.array([r[:,0],r[:,1],r[:,2],self.specie]).T)
    f = open("ATOMS.OUT","w")
    for i in range(len(r)):
        s = "S"+str(int(self.specie[i]))
        f.write(s+"   ")
        for ix in r[i]: f.write(str(ix)+"  ")
        f.write("\n")
    f.close()



def dict2array(d,ns=None):
    """Transform a dictionary into an array"""
    nn = max([di[2] for di in d]) # number of neighbors
    if ns is None:
      ns = max([di[0] for di in d]) # number of neighbors
    m = [np.zeros((ns,ns)) for i in range(nn)] # initialize
    for di in d:
        m[di[2]-1][di[0],di[1]] = d[di] # store value
        m[di[2]-1][di[1],di[0]] = d[di] # store value
    return np.array(m)


def array2dict(v):
    d = dict()
    for ni in range(len(v)):
        for i in range(len(v[ni])):
          for j in range(len(v[ni])):
              d[(i,j,ni+1)] = v[ni][i][j]
    return d



