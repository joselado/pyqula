#!/usr/bin/python
"""Remove the output files that calculations leave in the repository.

Only untracked files are removed, so a tracked file that happens to match a
pattern (MANIFEST.in) is kept. The .git and .claude folders are skipped.
"""

import fnmatch
import os
import subprocess

patterns = ["*.in","*.OUT","*.npz","*.pkl"]
skip = {".git",".claude"}

root = os.path.dirname(os.path.abspath(__file__))
tracked = set(subprocess.run(["git","ls-files"],cwd=root,check=True,
                             capture_output=True,text=True).stdout.split("\n"))

for dirpath,dirnames,filenames in os.walk(root):
    dirnames[:] = [d for d in dirnames if d not in skip]
    for f in filenames:
        if not any(fnmatch.fnmatch(f,p) for p in patterns): continue
        path = os.path.join(dirpath,f)
        if os.path.relpath(path,root) in tracked: continue
        os.remove(path)
