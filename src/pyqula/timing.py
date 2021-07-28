from __future__ import print_function,division
import time

printall = False

class Testimator:
  def __init__(self,title="",maxite=None,silent=False):
    self.t0 = time.perf_counter() # starting time
    self.title = title
    self.maxite = maxite
    self.silent = silent
    self.i = 0
    if len(title)>0 and not silent: print(title)
  def remaining(self,i,tot):
    """Print the ramining time in this task"""
    t = time.perf_counter() # current time
    dt = t - self.t0 # difference in time
    out = self.title + " " # empty line
    for j in range(10):
      if j<(i/tot*10): out += "#"
      else: out += " "
    out += str(round(i/tot*100))+"% completed,"
    trem = dt/(i+1)*(tot-i) # remaining time
    out += " remaining time "+str(round(trem,1))+"s"
    out += ", total time "+str(round(dt,1))+"s"
    if printall: print(out)
    else: print(out,end="\r")
  def iterate(self):
      if self.silent: return
      if self.maxite is not None: # of it has been provided
        self.remaining(self.i,self.maxite) # execute
        self.i += 1 # increase
