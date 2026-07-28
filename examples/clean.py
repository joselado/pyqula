import os
import glob
ds = os.walk(os.getcwd())

for d in ds:
  os.chdir(d[0])
  for name in glob.glob("*.OUT"): os.remove(name)

