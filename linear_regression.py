
# coding: utf-8

# In[11]:


import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn import linear_model
model = linear_model.LinearRegression()
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import numpy as np

# allow plots to appear directly in the notebook
get_ipython().magic('matplotlib inline')



# In[12]:


import glob

path = r'C:\Users\Dell\Downloads\BlogFeedback' # use your path
all_files = glob.glob(r'C:\Users\Dell\Downloads\BlogFeedback' + "/*.csv")

li = []

for BlogFeedback in all_files:
    data = pd.read_csv(BlogFeedback)
    li.append(data)

frame = pd.concat(li, axis=0)


# In[13]:


data.head()


# In[14]:


feature_cols = ['40.30467']
X = data[feature_cols]
feature1_cols = ['1.0.2']
y = data[feature1_cols]
# instantiate and fit
lm2 = LinearRegression()
lm2.fit(X, y)

# print the coefficients
print(lm2.intercept_)
print(lm2.coef_)


# In[17]:


-2.36074947 + 0.23135443*50


# In[18]:


X_new = pd.DataFrame({'40.30467': [50]})

# predict for a new observation
lm2.predict(X_new)

