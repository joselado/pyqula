import os
ds = os.walk(os.getcwd())


for d in ds:
  os.chdir(d) # go to that directory
  if os.path.isfile("main.py"):
      print("Running")
      print(d)
      os.system("python main.py")

