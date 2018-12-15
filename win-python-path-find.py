import os, sys

#change working directory
os.chdir("C:\\Users\\asus\\Desktop")

#find python Installed path
filename = os.path.dirname(sys.executable)

#redirect the output to text file

with open('out.txt', 'w+') as f:
	print("Filename:", filename, file=f)
	
	

# python.exe -m pip install --upgrade pip

