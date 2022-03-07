import os
ds = os.walk(os.getcwd())

ds = [d[0] for d in ds] # loop

for d in ds:
  os.chdir(d) # go to that directory
  if os.path.isfile("main.py"):
      print("Running")
      print(d)
      os.system("timeout 1s python main.py")

