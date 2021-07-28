from __future__ import print_function
import os

pairs = [] # pairs of folders, fortran file names and libraries


pairs += [("first_neigh","first_neighborsf90","first_neighborsf90")]
pairs += [("gauss_inv","gauss_inv","gauss_inv")]
pairs += [("kpm","kpmf90","kpmf90")]
pairs += [("mean_field","mean_fieldf90","mean_fieldf90")]

f2py = "f2py2.7" # compilator
flags = "-llapack -c -m"
for p in pairs:
  os.chdir(p[0])
  line = f2py+"  "+flags+"  "+p[1]+"   "+p[1]+".f90"
  os.system(line) # compile
  os.system("cp "+p[1]+".so  ../../"+p[2]+".so")
  os.chdir("..")


