#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics


# In[2]:


col_names = ['Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching', 'Irritability', 'delayed healing', 'partial paresis', 'muscle stiffness', 'Alopecia', 'Obesity', 'class']


# In[3]:


data = pd.read_csv("diabetes_data_upload.csv", header=None, names=col_names)


# In[4]:


print(data)


# In[5]:


attributes = ['Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching', 'Irritability', 'delayed healing', 'partial paresis', 'muscle stiffness', 'Alopecia', 'Obesity']
X = data[attributes]
y = data['class']


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# In[7]:


print(X_train)


# In[8]:


X_train = pd.get_dummies(X_train,drop_first=True)
X_test = pd.get_dummies(X_test,drop_first =True)


# In[9]:


print(X_train)


# In[10]:


print(X_test)


# In[11]:



final_train, final_test = X_train.align(X_test, join='inner', axis=1)


# In[12]:


print(final_train)
print(final_test)


# In[13]:


clf = DecisionTreeClassifier()
clf = clf.fit(final_train, y_train)


# In[14]:


y_pred = clf.predict(final_test)


# In[15]:


print(y_pred)


# In[16]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:




