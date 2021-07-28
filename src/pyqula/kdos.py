from __future__ import print_function,division
import numpy as np
from . import green
from . import dos
from . import klist
from . import kpm
from . import timing
from . import multicell
from . import kpm
from . import sculpt
from . import parallel
from . import algebra

def write_kdos(k=0.,es=[],ds=[],new=True):
  """ Write KDOS in a file"""
  if new: f = open("KDOS.OUT","w") # open new file
  else: f = open("KDOS.OUT","a") # continue writting
  for (e,d) in zip(es,ds): # loop over e and dos
    f.write(str(k)+"     ")
    f.write(str(e)+"     ")
    f.write(str(d)+"\n")
  f.close()





def kdos1d_sites(h,sites=[0],scale=10.,nk=100,npol=100,kshift=0.,
                  ewindow=None,info=False):
  """ Calculate kresolved density of states of
  a 1d system for a certain orbitals"""
  if h.dimensionality!=1: raise # only for 1d
  ks = np.linspace(0.,1.,nk) # number of kpoints
  h.turn_sparse() # turn the hamiltonian sparse
  hkgen = h.get_hk_gen() # get generator
  if ewindow is None:  xs = np.linspace(-0.9,0.9,nk) # x points
  else:  xs = np.linspace(-ewindow/scale,ewindow/scale,nk) # x points
  write_kdos() # initialize file
  for k in ks: # loop over kpoints
    mus = np.array([0.0j for i in range(2*npol)]) # initialize polynomials
    hk = hkgen(k+kshift) # hamiltonian
    for isite in sites:
      mus += kpm.local_dos(hk/scale,i=isite,n=npol)
    ys = kpm.generate_profile(mus,xs) # generate the profile
    write_kdos(k,xs*scale,ys,new=False) # write in file (append)
    if info: print("Done",k)

#
#def surface(h,energies=None,klist=None,delta=0.01):
#  """Return bulk and surface DOS"""
#  bout = [] # empty list, bulk
#  sout = [] # empty list, surface
#  for k in klist:
#    for energy in energies:
#      gs,sf = green.green_kchain(h,k=k,energy=energy,delta=delta,only_bulk=False) 
#      bout.append(gs.trace()[0,0].imag) # bulk
#      sout.append(sf.trace()[0,0].imag) # surface
#  bout = np.array(bout).reshape((len(energies),len(klist))) # convert to array
#  sout = np.array(sout).reshape((len(energies),len(klist))) # convert to array
#  return (bout.transpose(),sout.transpose())
#
#
#






def write_surface(h,energies=np.linspace(-.5,.5,300),
        klist=None,delta=None,operator=None,hs=None):
  if delta is None: delta = (np.max(energies)-np.min(energies))/len(energies)
  if h.dimensionality==1:
    write_surface_1d(h,energies=energies,delta=delta,
                         operator=operator)
  elif h.dimensionality==2:
    write_surface_2d(h,energies=energies,klist=klist,delta=delta,
                         operator=operator,hs=hs)
  elif h.dimensionality==3:
    write_surface_3d(h,energies=energies,klist=klist,delta=delta)
  else: raise




def write_surface_1d(h,energies=None,delta=None,
        operator=None):
  if energies is None: energies = np.linspace(-.5,.5,200)
  if delta is None: delta = (max(energies)-min(energies))/len(energies)
  h = h.get_no_multicell()
  fo  = open("SURFACE_DOS.OUT","w") # open file
  for energy in energies:
      gs,sf = green.green_renormalization(h.intra,h.inter,
              energy=energy,delta=delta) # surface green function 
      if operator is None: op = np.identity(h.intra.shape[0]) # identity matrix
      elif callable(operator): op = callable(op)
      else: op = operator # assume a matrix
      db = -algebra.trace(gs*op).imag # bulk
      ds = -algebra.trace(sf*op).imag # surface
      fo.write(str(energy)+"   "+str(ds)+"   "+str(db)+"\n")
      fo.flush()
  fo.close()





def write_surface_2d(h,energies=None,klist=None,delta=0.01,
                         operator=None,hs=None):
  bout = [] # empty list, bulk
  sout = [] # empty list, surface
  if klist is None: 
      klist = [[i,0.,0.] for i in np.linspace(-.5,.5,50)]
  if energies is None: energies = np.linspace(-.5,.5,50)
  fo  = open("KDOS.OUT","w") # open file
  for k in klist:
    print("Doing k-point",k)
    for energy in energies:
      gs,sf = green.green_kchain(h,k=k,energy=energy,delta=delta,
                       only_bulk=False,hs=hs) # surface green function 
      if operator is None: op = np.identity(h.intra.shape[0]) # identity matrix
      elif callable(operator): op = callable(op)
      else: op = operator # assume a matrix
      db = -algebra.trace(gs*op).imag # bulk
      ds = -algebra.trace(sf*op).imag # surface
      fo.write(str(k)+"   "+str(energy)+"   "+str(ds)+"   "+str(db)+"\n")
      fo.flush()
  fo.close()


def write_surface_3d(h,energies=None,klist=None,delta=0.01):
  raise # not implemented
  if h.dimensionality != 3: raise # only for 3d
  ho = h.copy() # copy Hamiltonian
  ho = ho.turn_multicell() # multicell Hamiltonian
  bout = [] # empty list, bulk
  sout = [] # empty list, surface
  if klist is None: raise
  if energies is None: energies = np.linspace(-.5,.5,50)
  fo  = open("KDOS.OUT","w") # open file
  for k in klist:
    for energy in energies:
      gs,sf = green.green_kchain(h,k=k,energy=energy,delta=delta,only_bulk=False) 
      db = -algebra.trace(gs).imag # bulk
      ds = -algebra.trace(sf).imag # surface
      fo.write(str(k)+"   "+str(energy)+"   "+str(ds)+"   "+str(db)+"\n")



def kdos_bands(h,use_kpm=False,kpath=None,scale=10.0,frand=None,
                 ewindow=4.0,ne=1000,delta=0.01,ntries=10,nk=100,
                 operator=None,energies=np.linspace(-3.0,3.0,200),
                 mode="ED",**kwargs):
  """Calculate the KDOS bands using the KPM"""
  if use_kpm: mode ="KPM" # conventional method
  if mode=="ED":
      from . import dos
      def pfun(k):
        (es,ds) = h.get_dos(ks=[k],operator=operator,energies=energies,
                delta=delta)
        return energies,ds
  elif mode=="green":
    f = h.get_gk_gen(delta=delta) # Green generator
    def pfun(k): # do it for this k-point
        def gfun(e):
            m = f(k=k,e=e) # Green's function
            m = green.GtimesO(m,operator,k=k)
            return -algebra.trace(m).imag # return DOS
        return energies,np.array([gfun(e) for e in energies])
  elif mode=="KPM": # KPM method
    if operator is not None: return NotImplemented # not implemented
    hkgen = h.get_hk_gen() # get generator
    def pfun(k): # do it for this k-point
      hk = hkgen(k) # get Hamiltonian
      npol = int(scale/delta) # number of polynomials
      (x,y) = kpm.tdos(hk,scale=scale,npol=npol,ne=ne,frand=frand,
                   ewindow=ewindow,ntries=ntries,x=energies) # compute
      return (x,y)
  if kpath is None: 
      kpath = klist.default(h.geometry,nk=nk) # default
  ### Now compute and write in a file
  ik = 0
  out = parallel.pcall(pfun,kpath) # compute all
  fo = open("KDOS_BANDS.OUT","w") # open file
  for k in kpath: # loop over kpoints
    (x,y) = out[ik] # get this one
    for (ix,iy) in zip(x,y): # loop
      fo.write(str(ik/len(kpath))+"   ")
      fo.write(str(ix)+"   ")
      fo.write(str(iy)+"\n")
    fo.flush()
    ik += 1
  fo.close()









def write_surface_kpm(h,ne=400,klist=None,scale=4.,npol=200,w=20,ntries=20):
  """Write the surface DOS using the KPM"""
  if klist is None: klist = np.linspace(-.5,.5,50)
  fo  = open("KDOS.OUT","w") # open file
  for k in klist:
    print("Doing kpoint",k)
    if h.dimensionality==2: 
      (intra,inter) = h.kchain(k) # k hamiltonian
      (es,ds,dsb) = kpm.edge_dos(intra,inter,scale=scale,w=w,npol=npol,
                            ne=ne,bulk=True)
    # if the Hamiltonian is 1d from the beginning
    elif h.dimensionality==1: 
      intra,inter = h.intra,h.inter # 1d hamiltonian
      dd = h.intra.shape[0] # dimension
      inde = np.zeros(dd) # array with zeros
      indb = np.zeros(dd) # array with zeros
      for i in range(dd//10): # one tenth
        inde[i] = 1. # use this one
        indb[4*dd//10 + i] = 1. # use this one
      def gedge(): return (np.random.random(len(inde))-0.5)*inde
      def gbulk(): return (np.random.random(len(indb))-0.5)*(indb)
      # hamiltonian
      h0 = intra + inter*np.exp(1j*np.pi*2.*k) + (inter*np.exp(1j*np.pi*2.*k)).H
      xs = np.linspace(-0.9,0.9,4*npol) # x points
      es = xs*scale
      # calculate the bulk
      mus = kpm.random_trace(h0/scale,ntries=ntries,n=npol,fun=gbulk)
      dsb = kpm.generate_profile(mus,xs) # generate the profile
      # calculate the edge
      mus = kpm.random_trace(h0/scale,ntries=ntries,n=npol,fun=gedge)
      ds = kpm.generate_profile(mus,xs) # generate the profile
    else: raise
    for (e,d1,d2) in zip(es,ds,dsb):
      fo.write(str(k)+"   "+str(e)+"   "+str(d1)+"    "+str(d2)+"\n")
  fo.close()




def interface(h1,h2,energies=np.linspace(-1.,1.,100),operator=None,
                    delta=None,kpath=None,dh1=None,dh2=None,nk=50):
  """Get the surface DOS of an interface"""
  from scipy.sparse import csc_matrix,bmat
  if delta is None:
      delta = 1*(max(energies) - min(energies))/len(energies)
  if kpath is None: 
    if h1.dimensionality==1:
      kpath = [[0.,0.,0.]]
    elif h1.dimensionality==3:
      g2d = h1.geometry.copy() # copy Hamiltonian
      g2d = sculpt.set_xy_plane(g2d)
      kpath = klist.default(g2d,nk=nk)
    elif h1.dimensionality==2:
      kpath = [[k,0.,0.] for k in np.linspace(0.,1.,nk)]
    else: raise
#  tr = timing.Testimator("KDOS") # generate object
#  tr.remaining(ik,len(kpath)) # generate object
  ik = 0
  h1 = h1.get_multicell() # multicell Hamiltonian
  h2 = h2.get_multicell() # multicell Hamiltonian
  def computek(ik):
    k = kpath[ik] # get this one
#    for energy in energies:
#  (b1,s1,b2,s2,b12) = green.interface(h1,h2,k=k,energy=energy,delta=delta)
#      out = green.interface(h1,h2,k=k,energy=energy,delta=delta)
    outs = green.interface_multienergy(h1,h2,k=k,energies=energies,
            delta=delta,dh1=dh1,dh2=dh2)
    outstr = ""
    for (energy,out) in zip(energies,outs):
      if operator is None: 
        op = np.identity(h1.intra.shape[0]*2) # normal cell
        ops = np.identity(h1.intra.shape[0]) # supercell 
#      elif callable(operator): op = callable(op)
      else:
        op = operator # normal cell 
        ops = bmat([[csc_matrix(operator),None],[None,csc_matrix(operator)]])
      # write everything
      outstr += str(ik)+"   "+str(energy)+"   "
      for g in out: # loop
        if g.shape[0]==op.shape[0]: d = -algebra.trace(g@op).imag # bulk
        else: d = -algebra.trace(g@ops).imag # interface
        outstr += str(d)+"   "
      outstr += "\n"
    return outstr
  out = parallel.pcall(computek,range(len(kpath))) # compute all
  fo = open("KDOS_INTERFACE.OUT","w")
  fo.write("# k, E, Bulk1, Surf1, Bulk2, Surf2, interface\n")
  for o in out: fo.write(o)
  fo.close()

#      fo.flush() # flush
#  fo.close()







def surface(h1,energies=np.linspace(-1.,1.,100),operator=None,
                    delta=0.01,kpath=None,hs=None):
  """Get the surface DOS of an interface"""
  from scipy.sparse import csc_matrix,bmat
  if kpath is None: 
    if h1.dimensionality==3:
      g2d = h1.geometry.copy() # copy Hamiltonian
      g2d = sculpt.set_xy_plane(g2d)
      kpath = klist.default(g2d,nk=len(energies))
    elif h1.dimensionality==2:
      kpath = [[k,0.,0.] for k in np.linspace(0.,1.,len(energies))]
    elif h1.dimensionality==1: kpath = [[0.,0.,0.0]] # one dummy point
    else: raise
  fo = open("KDOS.OUT","w")
  fo.write("# k, E, Surface, Bulk\n")
  tr = timing.Testimator("KDOS") # generate object
  ik = 0
  h1 = h1.get_multicell() # multicell Hamiltonian
  for k in kpath:
    tr.remaining(ik,len(kpath)) # generate object
    ik += 1
    outs = green.surface_multienergy(h1,k=k,energies=energies,delta=delta,hs=hs)
    for (energy,out) in zip(energies,outs):
      # write everything
      if h1.dimensionality==1: fo.write(str(energy)+"   ")
      else: fo.write(str(ik)+"   "+str(energy)+"   ")
      for g in out: # loop
        if operator is None: d = -algebra.trace(g).imag # only the trace 
        elif callable(operator): d = operator(g,k=k) # call the operator
        else:  d = -algebra.trace(g@operator).imag # assume it is a matrix
        fo.write(str(d)+"   ") # write in a file
      fo.write("\n") # next line
      fo.flush() # flush
  fo.close()
