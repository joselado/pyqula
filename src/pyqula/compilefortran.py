from __future__ import print_function
import os
import sys
import glob
import shutil
import subprocess

from . import filesystem as fs

# names is a lists with pairs of name of folder, f90 file and .so file

def compile_fortran(compiler=None):
    names = [("first_neigh","first_neighborsf90.f90","first_neighborsf90")] 
    names += [("kpm","kpm.f90","kpmf90")] 
    names += [("dos","dos.f90","dosf90")] 
    names += [("berry","berry_curvature.f90","berry_curvaturef90")] 
    names += [("gauss_inv","gauss_inv.f90","gauss_invf90")] 
    names += [("clean_geometry","clean_geometryf90.f90","clean_geometryf90")] 
    names += [("correlators","correlatorsf90.f90","correlatorsf90")]
    names += [("supercell","supercellf90.f90","supercellf90")] 
    names += [("classicalspin","classicalspinf90.f90","classicalspinf90")] 
    names += [("density_matrix","density_matrixf90.f90","density_matrixf90")] 
    names += [("kanemele","kanemelef90.f90","kanemelef90")]
    names += [("specialhopping","specialhoppingf90.f90","specialhoppingf90")]
    names += [("chi","chif90.f90","chif90")] 
    names += [("tails","tailsf90.f90","tailsf90")] 
    names += [("algebra","algebraf90.f90","algebraf90")] 
    names += [("dyson","dyson2d.f90","dyson2df90")] 
    
    if compiler is None:
      pybin = os.path.dirname(os.path.realpath(sys.executable))
      candidates = [os.path.join(pybin,"f2py"),os.path.join(pybin,"f2py.exe"),
                    os.path.join(pybin,"Scripts","f2py.exe")] # posix, and two common windows layouts
      compiler = next((c for c in candidates if os.path.isfile(c)),None)
      if compiler is None: compiler = shutil.which("f2py") # last resort, search PATH
    if compiler is None or not os.path.isfile(compiler):
        print("f2py compiler not found")
        print("You may want to use another Python")
        exit()

    print("Using compiler",compiler)
    dirname = os.path.dirname(os.path.realpath(__file__)) # main folder
    pwd = os.getcwd() # current directory

    for name in names:
      folder,f90,so = name[0],name[1],name[2] # different names
      os.chdir(os.path.join(dirname,"fortran",folder)) # go to the folder
      fs.rmglob("*.so") ; fs.rmglob("*.pyd") # remove old libraries
      print("Compiling",name[1])
      subprocess.run([compiler,"-c","-m",so,f90], # compile
                      stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
      built = glob.glob(so+"*.so") + glob.glob(so+"*.pyd") # compiled extension
      if built: fs.cpfile(built[0],os.path.join(dirname,so+os.path.splitext(built[0])[1])) # copy library
    os.chdir(pwd) # return to parent directory

if __name__=="__main__":
    compile_fortran()
