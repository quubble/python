#File reading and writing without fuction

import sys, os

#change directory path
os.chdir("C:\\Users\\asus\\Desktop")

file = open('a.txt' ,'w+') 
file.write('Hello Python\n') 
file.write('This is our new Python text file.\n') 
file.write('This is another line in Python file.\n') 
file.write('Why? Because we can add as many lines as we want to this Python file.\n')

#reading the file 1st time
file.seek(0)       
r = file.read()
print(r)

#reading the file 2nd time, each time set "seek(0)" before reading.
file.seek(0)
r = file.read()
print(r)

file.close()