from __future__ import print_function
import os
import glob
import shutil
import subprocess

pairs = [] # pairs of folders, fortran file names and libraries


pairs += [("first_neigh","first_neighborsf90","first_neighborsf90")]
pairs += [("gauss_inv","gauss_inv","gauss_inv")]
pairs += [("kpm","kpmf90","kpmf90")]
pairs += [("mean_field","mean_fieldf90","mean_fieldf90")]

f2py = "f2py2.7" # compilator
flags = ["-llapack","-c","-m"]
for p in pairs:
  os.chdir(p[0])
  subprocess.run([f2py]+flags+[p[1],p[1]+".f90"]) # compile
  built = glob.glob(p[1]+"*.so") + glob.glob(p[1]+"*.pyd")
  if built: shutil.copyfile(built[0],os.path.join("..","..",p[2]+os.path.splitext(built[0])[1]))
  os.chdir("..")


