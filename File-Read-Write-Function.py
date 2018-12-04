#write the output of a function to a text file in python

import sys, os

#change directory path
os.chdir("C:\\Users\\asus\\Desktop")

def out_fun():
    mystr = '''Hello World.
This is our new text file.
This is another line.
Why? Because we can add as many lines as we want to this file.

'''
    return mystr

output = out_fun()
file = open('testfilefunction.txt' ,'w+')
file.write(output)

#reading file contents.
file.seek(0)
x = file.read()
print(x)

file.close()