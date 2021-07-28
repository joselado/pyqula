#!/usr/bin/python


import os


pwd = os.getcwd()

fs = ["*.in","*.OUT","*.npz","*.pkl"]

for d in os.walk("."): # loop over subdirectories
  os.chdir(d[0])
  for f in fs:
    os.system("rm -rf "+f)
  os.chdir(pwd)

