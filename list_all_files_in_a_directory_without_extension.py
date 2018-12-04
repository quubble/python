

## How to list all files in a directory without extension in Python

import os, sys    
files_no_ext = [".".join(f.split(".")[:-1]) for f in os.listdir() if os.path.isfile(f)]
print(files_no_ext)


#List ('.') current directory

import sys, os
path = '.'
 
files = os.listdir(path)
for name in files:
    print(name)
	


#List any directory

import os,sys
 
path = '.'
 
if len(sys.argv) == 2:
    path = sys.argv[1]
 
 
files = os.listdir(path)
for name in files:
    print(name)

full_path = os.path.join(path, name)
print(full_path)

