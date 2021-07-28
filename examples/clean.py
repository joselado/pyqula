import os
ds = os.walk(os.getcwd())

for d in ds:
  os.chdir(d[0])
  os.system("rm -f *.OUT")

