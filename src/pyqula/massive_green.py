# library to perform massive green function calculations
import os
import numpy as np

from . import filesystem as fs

prec = 4



def round_folder(input_folder,output_folder,prec=prec):
  """Round the names of files in a certain folder"""
  names = os.listdir(os.getcwd()+"/"+input_folder)
  fs.rmdir(output_folder)
  fs.mkdir(output_folder)
  for ni in names:
    n = ni.split("_")
    pre = n[0]
    value = n[1].split(".dat")[0] # value
    value = round(float(value),prec) # value with new precision
    name = pre + "_"+str(value)+".dat"
    os.system("cp "+input_folder+"/"+ni+"  "+output_folder+"/"+name) # copy to the new




def clean():
  """ Clena all the directories"""
  os.system("rm -rf green_storage*")
  os.system("rm -rf pdos_storage*")


def get_green(fun_gf,name="",energy=0.0,prec=prec):
  """Looks for the green function at that energy, if it hasn't been calculated
  calculates it"""
  foldername = "green_storage_"+name # name of the foler
  dirname = os.getcwd()+"/"+foldername # name of the foler
  if not foldername in os.listdir(os.getcwd()):
    fs.mkdir(foldername) # create the folder if nonexistent
  er = round(energy,prec) # round the energy value
  namefile = "green_"+str(er)+".npy" # name of the file
  files = os.listdir(dirname) # files in the directory
  if namefile in files: # check if it has been calculated
    m = np.matrix(np.load(dirname+"/"+namefile)) # get the matrix
    return m # return the matrix
  else: # if it hasn't been calculated
    m = fun_gf(er) # calcualte green function
    np.save(dirname+"/"+namefile,m) # save the matrix
    return m # return the matrix



def get_pdos(fun_pd,name="",energy=0.0,prec=6):
  """Looks for the DOS at that energy, if it hasn't been calculated
  calculates it"""
  foldername = "pdos_storage_"+name # name of the foler
  dirname = os.getcwd()+"/"+foldername # name of the foler
  if not foldername in os.listdir(os.getcwd()):
    fs.mkdir(foldername) # create the folder if nonexistent
  er = round(energy,prec) # round the energy value
  namefile = "pdos_"+str(er)+".dat" # name of the file
  files = os.listdir(dirname) # files in the directory
  if namefile in files: # check if it has been calculated
    m = np.loadtxt(dirname+"/"+namefile) # get the matrix
    return m # return the matrix
  else: # if it hasn't been calculated
    m = fun_pd(er) # calcualte green function
    np.savetxt(dirname+"/"+namefile,m) # save the matrix
    return m # return the matrix


