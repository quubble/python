#!/usr/bin/env python
# coding: utf-8

# In[73]:


mylist=[1,2,3,4,5,6,7,8,9,10]
x = map(lambda x: x**0, mylist)
list(x)http://127.0.0.1:8888/notebooks/Untitled5.ipynb?kernel_name=python3#


# In[31]:


mylist=[1,2,3,4,5,6,7,8,9,10]
x = filter(lambda x: x**3, mylist)
print(list(x))


# In[33]:


# Filter list of numbers by keeping numbers from 10 to 20 in the list only

listofNum = [1,3,33,12,34,56,11,19,21,34,15]
listofNum = list(filter(lambda x : x > 10 and x < 20, listofNum))
print('Filtered List : ', listofNum)


# In[80]:


import math

mylist=[1,2,3,4,5,2,4,6,8]
x =map(lambda x: (math.pow(x,2)),mylist)

print("The list is: " + str(list(x)))


# In[81]:


list(filter(lambda x:(x % 2 == 0),mylist))


# In[83]:


list(map(lambda x:(math.pow(x,2)),mylist))


# In[104]:


mystring =["Shankar","Brahama","Vishnu","Mahesh","Saraswati","Laxmi","Parvati"]
list(mystring)


# In[105]:


list(map(lambda s:s[::-len(s)],mystring))


# In[106]:


list(map(lambda s:s[::-1],mystring))


# In[107]:


def palindrome(s):
    
    s = s.replace(' ','') # This replaces all spaces " " with no space ''. (Fixes issues with strings that have spaces)
    return s == s[::-1] # Check through slicing


# In[110]:


palindrome('nurseskkk')


# In[ ]:




