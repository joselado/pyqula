import os
import sys
import subprocess
ds = os.walk(os.getcwd())

ds = [d[0] for d in ds] # loop

for d in ds:
  os.chdir(d) # go to that directory
  if os.path.isfile("main.py"):
      print("Running")
      print(d)
      try: subprocess.run([sys.executable,"main.py"],timeout=1)
      except subprocess.TimeoutExpired: pass
