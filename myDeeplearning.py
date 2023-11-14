#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import seaborn as sns

import tensorflow as tf
df = pd.read_csv('https://raw.githubusercontent.com/donghaeK/dataSet/main/myDataset.csv')


# In[2]:


X = df[['grade','study_Time','salary','Certification','Positive']]
#X = df.iloc[:,[0,1,2,3,4]]
#rint(X)
#rint(type(X))


# In[3]:


y = df['Salary after 10 years']
#rint(type(y))


# In[4]:


#print(X.values)
#print(y.values)


# In[5]:


regr = linear_model.LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)


# In[6]:


print(y_pred)
#print(y_test)


# In[7]:


score = regr.score(X_train, y_train)
print(score)
score = regr.score(X_test, y_test)
print(score)


# In[8]:


plt.scatter(y_pred, y_test, color='blue')
x=np.linspace(40,65,100)
plt.plot(x,x, linewidth=5, color='black')
plt.show()


# In[9]:


sns.set(rc={'figure.figsize':(10,10)})
correlation_matrix = df.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)
plt.show()


# In[ ]:




