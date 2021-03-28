#!/usr/bin/env python
# coding: utf-8

# # Data Mining

# ## Midterm Project

# ## Apriori Algorithm

# ### Submitted by - Parth Sharma (UCID - 31543562) Mail - ps75@njit.edu

# ##### How to run: - Jupyter Notebook will be the best way to run this program. This program uses the name and the address of the CSV file which is present in my current system. To avoid error, make sure you have CSV file with same name and path, or you can change the attributes in the program accordingly.
# 

# ###### About the Program: This program shows the comparison between the 2 approaches which are used to find out the frequent item set. This program takes the input from user at the run time input such as
# • Minimum support
# 
# • Minimum Confidence
# 

# ### Code and the output is given below.

# ### Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import itertools


# In[2]:


i = int(input('Enter any desired value '))
if i == 1:
    data = pd.read_csv('GroceryStoreDataSet.csv')
if i ==2:
    data = pd.read_csv('nike.csv')
else:
    data = pd.read_csv('flipkart.csv')


# ## Dataset 1 
# 
# Note- (i=1 for dataset 1)

# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.shape


# ## List of all unique items in each transaction 

# In[6]:


for i in range(data.shape[0]):
    print(f'List of Unique Items in Transaction : {i} ')
    print(data.iloc[i].dropna().unique(),' \n')


# In[7]:


print(f'Total no. of transactions in this Nike set {data.shape[0]}')


# In[8]:


minimum_s_count = float(input('enter any value for min support count '))


# In[9]:


minimum_confi = float(input('Enter any desired value for confidence '))


# In[10]:


records = []
for i in range(0, data.shape[0]):
    records.append([str(data.values[i,j]) for j in range(0, data.shape[1])])

items = sorted([item for sublist in records for item in sublist if item != 'nan'])


# ### List of Unique itemsets in this dataset

# In[11]:


set(items)


# In[12]:


# To get a single item with their overall count in all transactions given.

def stage_1(items, minimum_s_count):
    c1 = {i:items.count(i) for i in items}
    l1 = {}
    for key, value in c1.items():
        if value >= minimum_s_count:
            l1[key] = value 
    
    return c1, l1


# In[13]:


# To get sets of 2 items with their overall count in all transactions given.
def stage_2(l1, records, minimum_s_count):
    l1 = sorted(list(l1.keys()))
    L1 = list(itertools.combinations(l1, 2))              # To make all possible combinations of 2 items
    c2 = {}
    l2 = {}
    for iter1 in L1:
        count = 0
        for iter2 in records:
            if sublist(iter1, iter2):
                count+=1
        c2[iter1] = count
    for key, value in c2.items():
        if value >= minimum_s_count:
            if check_subset_frequency(key, l1, 1):
                l2[key] = value 
    
    return c2, l2


# In[14]:


# To get sets of 3 items with their overall count in all transactions given.    
def stage_3(l2, records, minimum_s_count):
    l2 = list(l2.keys())
    L2 = sorted(list(set([item for t in l2 for item in t])))
    L2 = list(itertools.combinations(L2, 3))           # To make all possible combinations of 3 items
    c3 = {}
    l3 = {}
    for iter1 in L2:
        count = 0
        for iter2 in records:
            if sublist(iter1, iter2):
                count+=1
        c3[iter1] = count
    for key, value in c3.items():
        if value >= minimum_s_count:
            if check_subset_frequency(key, l2, 2):
                l3[key] = value 
        
    return c3, l3


# In[15]:


# To get sets of 4 items with their overall count in all transactions given.
def stage_4(l3, records, minimum_s_count):
    l3 = list(l3.keys())
    L3 = sorted(list(set([item for t in l3 for item in t])))
    L3 = list(itertools.combinations(L3, 4))             # To make all possible combinations of 4 items
    c4 = {}
    l4 = {}
    for iter1 in L3:
        count = 0
        for iter2 in records:
            if sublist(iter1, iter2):
                count+=1
        c4[iter1] = count
    for key, value in c4.items():
        if value >= minimum_s_count:
            if check_subset_frequency(key, l3, 3):
                l4[key] = value 
        
    return c4, l4


# In[16]:


def sublist(lst1, lst2):
    return set(lst1) <= set(lst2)
    
def check_subset_frequency(itemset, l, n):
    if n>1:    
        subsets = list(itertools.combinations(itemset, n))
    else:
        subsets = itemset
    for iter1 in subsets:
        if not iter1 in l:
            return False
    return True


# ## Below part prints all L1,L2,L3
# 
# ###### All of them have pairs of 1 item, 2 items, 3 items respectively following minimum support decided by user above.

# In[17]:


c1, l1 = stage_1(items, minimum_s_count)
c2, l2 = stage_2(l1, records, minimum_s_count)
c3, l3 = stage_3(l2, records, minimum_s_count)
c4, l4 = stage_4(l3, records, minimum_s_count)


# In[18]:


for key, value in l1.items():
    print(key, ' : ', value)


# In[19]:


for key, value in l2.items():
    print(key, ' : ', value)


# In[20]:


for key, value in l3.items():
    print(key, ' : ', value)


# ## Print Association Rules
# 
# ###### (with their Minimum support & minimum confidence)

# In[21]:


itemlist = {**l1, **l2, **l3, **l4}

def support_count(itemset, itemlist):
    return itemlist[itemset]

def print_sets():
    print('Possible sets with just 1 item ',l1)

sets = []
for iter1 in list(l3.keys()):
    subsets = list(itertools.combinations(iter1, 2))
    sets.append(subsets)

list_l3 = list(l3.keys())
print('ASSOCIATION RULES : '+ '\n')
for i in range(0, len(list_l3)):
    for iter1 in sets[i]:
        a = iter1
        b = set(list_l3[i]) - set(iter1)
        confidence = (support_count(list_l3[i], itemlist)/support_count(iter1, itemlist))*100
        support_i = support_count(iter1,itemlist)
        if confidence > minimum_confi:
            print("Confidence {}->{} = ".format(a,b), round(confidence,3),'%')
            print('Support{}->{} = '.format(a,b),round(support_i,3))
            print('\n')
#         print("Confidence{}->{} = ".format(a,b), confidence)


# # Dataset 2
# Note- (i=2 for dataset 2)
# 

# In[22]:


i = int(input('Enter any desired value '))
if i == 1:
    data = pd.read_csv('GroceryStoreDataSet.csv')
if i ==2:
    data = pd.read_csv('nike.csv')
else:
    data = pd.read_csv('flipkart.csv')


# In[23]:


data.head()


# In[24]:


data.tail()


# In[25]:


data.shape


# ## List of all unique items in each transaction

# In[26]:


for i in range(data.shape[0]):
    print(f'List of Unique Items in Transaction : {i} ')
    print(data.iloc[i].dropna().unique(),' \n')


# In[27]:


print(f'Total no. of transactions in this Nike set {data.shape[0]}')


# In[28]:


minimum_s_count = float(input('enter any value for min support count '))


# In[29]:


minimum_confi = float(input('Enter any desired value for confidence '))


# In[30]:


records = []
for i in range(0, data.shape[0]):
    records.append([str(data.values[i,j]) for j in range(0, data.shape[1])])

items = sorted([item for sublist in records for item in sublist if item != 'nan'])


# ## List of Unique itemsets in this dataset

# In[31]:


set(items)


# In[32]:


# To get a single item with their overall count in all transactions given.

def stage_1(items, minimum_s_count):
    c1 = {i:items.count(i) for i in items}
    l1 = {}
    for key, value in c1.items():
        if value >= minimum_s_count:
            l1[key] = value 
    
    return c1, l1


# In[33]:


# To get sets of 2 items with their overall count in all transactions given.
def stage_2(l1, records, minimum_s_count):
    l1 = sorted(list(l1.keys()))
    L1 = list(itertools.combinations(l1, 2))              # To make all possible combinations of 2 items
    c2 = {}
    l2 = {}
    for iter1 in L1:
        count = 0
        for iter2 in records:
            if sublist(iter1, iter2):
                count+=1
        c2[iter1] = count
    for key, value in c2.items():
        if value >= minimum_s_count:
            if check_subset_frequency(key, l1, 1):
                l2[key] = value 
    
    return c2, l2


# In[34]:


# To get sets of 3 items with their overall count in all transactions given.    
def stage_3(l2, records, minimum_s_count):
    l2 = list(l2.keys())
    L2 = sorted(list(set([item for t in l2 for item in t])))
    L2 = list(itertools.combinations(L2, 3))           # To make all possible combinations of 3 items
    c3 = {}
    l3 = {}
    for iter1 in L2:
        count = 0
        for iter2 in records:
            if sublist(iter1, iter2):
                count+=1
        c3[iter1] = count
    for key, value in c3.items():
        if value >= minimum_s_count:
            if check_subset_frequency(key, l2, 2):
                l3[key] = value 
        
    return c3, l3


# In[35]:


# To get sets of 4 items with their overall count in all transactions given.
def stage_4(l3, records, minimum_s_count):
    l3 = list(l3.keys())
    L3 = sorted(list(set([item for t in l3 for item in t])))
    L3 = list(itertools.combinations(L3, 4))             # To make all possible combinations of 4 items
    c4 = {}
    l4 = {}
    for iter1 in L3:
        count = 0
        for iter2 in records:
            if sublist(iter1, iter2):
                count+=1
        c4[iter1] = count
    for key, value in c4.items():
        if value >= minimum_s_count:
            if check_subset_frequency(key, l3, 3):
                l4[key] = value 
        
    return c4, l4


# In[36]:


def sublist(lst1, lst2):
    return set(lst1) <= set(lst2)
    
def check_subset_frequency(itemset, l, n):
    if n>1:    
        subsets = list(itertools.combinations(itemset, n))
    else:
        subsets = itemset
    for iter1 in subsets:
        if not iter1 in l:
            return False
    return True


# ## Below part prints all L1,L2,L3
# 
# ##### All of them have pairs of 1 item, 2 items, 3 items respectively following minimum support decided by user above

# In[37]:


c1, l1 = stage_1(items, minimum_s_count)
c2, l2 = stage_2(l1, records, minimum_s_count)
c3, l3 = stage_3(l2, records, minimum_s_count)
c4, l4 = stage_4(l3, records, minimum_s_count)


# In[38]:


for key, value in l1.items():
    print(key, ' : ', value)


# In[39]:


for key, value in l2.items():
    print(key, ' : ', value)


# In[40]:


for key, value in l3.items():
    print(key, ' : ', value)


# ## Print Association Rules
# 
# ###### (with their Minimum support & minimum confidence)

# In[41]:


itemlist = {**l1, **l2, **l3, **l4}

def support_count(itemset, itemlist):
    return itemlist[itemset]

def print_sets():
    print('Possible sets with just 1 item ',l1)

sets = []
for iter1 in list(l3.keys()):
    subsets = list(itertools.combinations(iter1, 2))
    sets.append(subsets)

list_l3 = list(l3.keys())
print('ASSOCIATION RULES : '+ '\n')
for i in range(0, len(list_l3)):
    for iter1 in sets[i]:
        a = iter1
        b = set(list_l3[i]) - set(iter1)
        confidence = (support_count(list_l3[i], itemlist)/support_count(iter1, itemlist))*100
        support_i = support_count(iter1,itemlist)
        if confidence > minimum_confi:
            print("Confidence {}->{} = ".format(a,b), round(confidence,3),'%')
            print('Support{}->{} = '.format(a,b),round(support_i,3))
            print('\n')
#         print("Confidence{}->{} = ".format(a,b), confidence)


# ## Dataset 3
# 
# Note- (Select any value for 'i' other than 1 & 2)

# In[42]:


i = int(input('Enter any desired value '))
if i == 1:
    data = pd.read_csv('GroceryStoreDataSet.csv')
if i ==2:
    data = pd.read_csv('nike.csv')
else:
    data = pd.read_csv('flipkart.csv')


# In[43]:


data.head()


# In[44]:


data.tail()


# In[45]:


data.shape


# ## List of all unique items in each transaction

# In[46]:


for i in range(data.shape[0]):
    print(f'List of Unique Items in Transaction : {i} ')
    print(data.iloc[i].dropna().unique(),' \n')


# In[47]:


print(f'Total no. of transactions in this dataset {data.shape[0]}')


# In[48]:


minimum_s_count = float(input('enter any value for min support count '))


# In[49]:


minimum_confi = float(input('Enter any desired value for confidence '))


# In[50]:


records = []
for i in range(0, data.shape[0]):
    records.append([str(data.values[i,j]) for j in range(0, data.shape[1])])

items = sorted([item for sublist in records for item in sublist if item != 'nan'])


# ### List of all unique items in this dataset

# In[51]:


set(items)


# In[52]:


# To get a single item with their overall count in all transactions given.

def stage_1(items, minimum_s_count):
    c1 = {i:items.count(i) for i in items}
    l1 = {}
    for key, value in c1.items():
        if value >= minimum_s_count:
            l1[key] = value 
    
    return c1, l1


# In[53]:


# To get sets of 2 items with their overall count in all transactions given.
def stage_2(l1, records, minimum_s_count):
    l1 = sorted(list(l1.keys()))
    L1 = list(itertools.combinations(l1, 2))              # To make all possible combinations of 2 items
    c2 = {}
    l2 = {}
    for iter1 in L1:
        count = 0
        for iter2 in records:
            if sublist(iter1, iter2):
                count+=1
        c2[iter1] = count
    for key, value in c2.items():
        if value >= minimum_s_count:
            if check_subset_frequency(key, l1, 1):
                l2[key] = value 
    
    return c2, l2


# In[54]:


# To get sets of 3 items with their overall count in all transactions given.    
def stage_3(l2, records, minimum_s_count):
    l2 = list(l2.keys())
    L2 = sorted(list(set([item for t in l2 for item in t])))
    L2 = list(itertools.combinations(L2, 3))           # To make all possible combinations of 3 items
    c3 = {}
    l3 = {}
    for iter1 in L2:
        count = 0
        for iter2 in records:
            if sublist(iter1, iter2):
                count+=1
        c3[iter1] = count
    for key, value in c3.items():
        if value >= minimum_s_count:
            if check_subset_frequency(key, l2, 2):
                l3[key] = value 
        
    return c3, l3


# In[55]:


# To get sets of 4 items with their overall count in all transactions given.
def stage_4(l3, records, minimum_s_count):
    l3 = list(l3.keys())
    L3 = sorted(list(set([item for t in l3 for item in t])))
    L3 = list(itertools.combinations(L3, 4))             # To make all possible combinations of 4 items
    c4 = {}
    l4 = {}
    for iter1 in L3:
        count = 0
        for iter2 in records:
            if sublist(iter1, iter2):
                count+=1
        c4[iter1] = count
    for key, value in c4.items():
        if value >= minimum_s_count:
            if check_subset_frequency(key, l3, 3):
                l4[key] = value 
        
    return c4, l4


# In[56]:


def sublist(lst1, lst2):
    return set(lst1) <= set(lst2)
    
def check_subset_frequency(itemset, l, n):
    if n>1:    
        subsets = list(itertools.combinations(itemset, n))
    else:
        subsets = itemset
    for iter1 in subsets:
        if not iter1 in l:
            return False
    return True


# ## Below part prints all L1,L2,L3
# ###### All of them have pairs of 1 item, 2 items, 3 items respectively following minimum support decided by user above

# In[57]:


c1, l1 = stage_1(items, minimum_s_count)
c2, l2 = stage_2(l1, records, minimum_s_count)
c3, l3 = stage_3(l2, records, minimum_s_count)
c4, l4 = stage_4(l3, records, minimum_s_count)


# In[58]:


for key, value in l1.items():
    print(key, ' : ', value)


# In[59]:


for key, value in l2.items():
    print(key, ' : ', value)


# In[60]:


for key, value in l3.items():
    print(key, ' : ', value)


# ## Print Association Rules
# 
# ###### (with their Minimum support & minimum confidence)

# In[61]:


itemlist = {**l1, **l2, **l3, **l4}

def support_count(itemset, itemlist):
    return itemlist[itemset]

def print_sets():
    print('Possible sets with just 1 item ',l1)

sets = []
for iter1 in list(l3.keys()):
    subsets = list(itertools.combinations(iter1, 2))
    sets.append(subsets)

list_l3 = list(l3.keys())
print('ASSOCIATION RULES : '+ '\n')
for i in range(0, len(list_l3)):
    for iter1 in sets[i]:
        a = iter1
        b = set(list_l3[i]) - set(iter1)
        confidence = (support_count(list_l3[i], itemlist)/support_count(iter1, itemlist))*100
        support_i = support_count(iter1,itemlist)
        if confidence > minimum_confi:
            print("Confidence {}->{} = ".format(a,b), round(confidence,3),'%')
            print('Support{}->{} = '.format(a,b),round(support_i,3))
            print('\n')
#         print("Confidence{}->{} = ".format(a,b), confidence)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




