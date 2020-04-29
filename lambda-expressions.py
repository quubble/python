#!/usr/bin/env python
# coding: utf-8

# In[1]:


def square(num):
    result = num**2
    return result


# In[2]:


square(2)


# In[3]:


def square(num):
    return num**2


# In[4]:


# We can actually write this in one line (although it would be bad style to do so)

def square(num): return num**2


# In[5]:



square(2)


# In[6]:


# This is the form a function that a lambda expression intends to replicate. A lambda expression can then be written as:
lambda num: num**2


# In[7]:


# Note how we get a function back. We can assign this function to a label:

square = lambda num: num**2
square(2)


# In[8]:


# Lambda: Check it a number is even

even = lambda x: x%2==0


# In[9]:


even(3)


# In[10]:


even(4)


# In[11]:


# Grab first character of a string:

first = lambda s: s[0]

first('hello')


# In[12]:


# Reverse a string

reverse = lambda s: s[::-1]

reverse('hello world!')


# In[13]:


# Just like a normal function, we can accept more than one function into a lambda expression:

adder = lambda x,y,z,p,q,r : [x+(y**z) + (p**q)/r]
adder(1,2,4,1,2,1)


# In[14]:


import math

def sqroot(x):
    """
    Finds the square root of the number passed in
    """
    return math.sqrt(x)


# In[15]:


sqroot(9)


# In[16]:


sqroot(64)


# In[17]:


square_rt = lambda x: math.sqrt(x)


# In[18]:


square_rt(144)


# In[19]:


import math

def topow(x,y):
    """
    Finds the square root of the number passed in
    """
    return math.pow(x,y)


# In[21]:


topow(2,3)


# In[22]:


import math

a = (lambda x,y: math.pow(x,y))
a (4,3)


# In[23]:


sequences = [10,2,8,7,5,4,3,11,0, 1]
filtered_result = map (lambda x: x*x, sequences) 
print(list(filtered_result))


# In[24]:


#What a lambda returns
string='some kind of a useless lambda'
print(lambda string : print(string))


# In[25]:


#What a lambda returns #2
x="some kind of a useless lambda"
(lambda x : print(x))(x)


# In[ ]:





# In[26]:


#A REGULAR FUNCTION
def guru( funct, *args ):
    funct( *args )
    
def printer_one( arg ):
    return print (arg)

def printer_two( arg ):
    print(arg)
    
#CALL A REGULAR FUNCTION 
guru( printer_one, 'printer 1 REGULAR CALL' )

guru( printer_two, 'printer 2 REGULAR CALL \n' )

#CALL A REGULAR FUNCTION THRU A LAMBDA
guru(lambda: printer_one('printer 1 LAMBDA CALL'))

guru(lambda: printer_two('printer 2 LAMBDA CALL'))


# In[27]:


(lambda x: x + x)(2)


# In[28]:


# lambda with filter function

sequences = [10,2,8,7,5,4,3,11,0, 1]
filtered_result = filter (lambda x: x < 4, sequences) 
print(list(filtered_result))


# In[29]:


# lambda with map function

sequences = [10,2,8,7,5,4,3,11,0, 1]
filtered_result = map (lambda x: x**x, sequences) 
print(list(filtered_result))


# In[30]:


x


# In[37]:


# lambda with map factorial function 

sequences = [100,2,6,8,7,5,4,3,11,0,9]

filtered_result = map(lambda x: 1 if x == 0 else x * (x-1), sequences)

print(list((filtered_result)))


# In[39]:


fact = lambda x: 1 if x == 0 else x * fact(x-1)
fact(11)


# In[34]:


# lambdas in reduce() here is a program that returns the product of all elements in a list:

""" Step 1) Perform the defined operation on the first 2 elements of the sequence.

Step 2) Save this result

Step 3) Perform the operation with the saved result and the next element in the sequence.

Step 4) Repeat until no more elements are left.

It also takes two parameters:

A function that defines the operation to be performed
A sequence (any iterator like lists, tuples, etc.)   """

import random
from functools import reduce

Start = 1
Stop = 11
limit = 5
# generate random numbers
sequences =[random.randrange(Start, Stop) for iter in range(limit)] 
print(sequences)

product = reduce(lambda x, y: x*y, sequences)
print(product)


# In[35]:


import random

Start = 9
Stop = 99
limit = 10
[random.randrange(Start, Stop) for iter in range(limit)]


# In[ ]:





# In[ ]:




