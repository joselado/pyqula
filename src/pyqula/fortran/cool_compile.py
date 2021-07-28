import os

#directory = os.getcwd() # current direction
#fols = [x[0] for x in os.walk(directory)] # get subdirectories
fols = next(os.walk('.'))[1] # subdirectories
import glob
for f in fols:
  src90 = glob.glob(f+'/*.f90')
  try:
    namef90 = src90[0].split("/")[-1] # get the name
    name = namef90.split(".")[0]
    os.system("f2py -llapack -c -m "+name +"  "+src90[0])
 #   print name
  except: pass
exit()
#f2py -llapack -c -m gauss_inv gauss_inv.f90
#f2py -llapack -c -m green_fortran green_fortran.f90
#cp *.so ../  # copy to the previous folder
