import os
ds = os.walk(os.getcwd())



def modify(ls):
    """Modify an input file to add the path"""
    ls = ls.split("\n") # split
    lo = "# Add the root path of the pyqula library\n"
    lo += "import os ; import sys \n"
    lo += 'sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")\n'
    for l in ls: # loop over lines
        if "sys.path.append" in l: continue
        if "import os ; import sys" in l: continue
        if "Add the root path" in l: continue
        else: lo += l + "\n" # add line
    return lo

ds = [d[0] for d in ds] # loop

for d in ds:
  os.chdir(d) # go to that directory
  if os.path.isfile("main.py"):
      ls = open("main.py").read() # read all the lines
      print(d)
      open("main.py","w").write(modify(ls)) # write file
#  os.system("rm -f *.OUT")

