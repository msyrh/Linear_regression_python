#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# 
# # Linear Regression Example
# 
# This example uses the only the first feature of the `diabetes` dataset, in
# order to illustrate a two-dimensional plot of this regression technique. The
# straight line can be seen in the plot, showing how linear regression attempts
# to draw a straight line that will best minimize the residual sum of squares
# between the observed responses in the dataset, and the responses predicted by
# the linear approximation.
# 
# The coefficients, the residual sum of squares and the variance score are also
# calculated.
# 
# 
# 

# In[12]:


print(__doc__)


# Code source: Jaques Grobler
# License: BSD 3 clause


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes = datasets.load_diabetes()


# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()


# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[23]:


USAhousing = pd.read_csv(r'D:\path file anda\train.csv') 
USAhousing=USAhousing.dropna()


# In[49]:


USAhousing


# In[43]:



#USAhousing['Gender'] =(USAhousing['Gender'] == 'Female').astype(float)
USAhousing['Property_Area'] =(USAhousing['Property_Area'] == 'Semiurban').astype(float)
USAhousing['Self_Employed'] =(USAhousing['Self_Employed'] == 'Yes').astype(float)
USAhousing['Married'] =(USAhousing['Married'] == 'Yes').astype(float)
USAhousing['Education'] =(USAhousing['Education'] == 'Graduate').astype(float)
USAhousing['Dependents'] =(USAhousing['Dependents'] == '3+').astype(float)


# In[44]:


USAhousing.info()


# In[45]:


USAhousing.describe()


# In[46]:


USAhousing.columns


# In[47]:


sns.pairplot(USAhousing)


# In[95]:


USAhousing['Credit_History'].hist(bins=50)


# In[94]:


USAhousing.boxplot(column='Credit_History')


# In[96]:


USAhousing['Gender'].hist(bins=50)


# In[97]:


USAhousing.boxplot(column='Gender')


# In[98]:


USAhousing['LoanAmount'].hist(bins=50)


# In[100]:


USAhousing.boxplot(column='LoanAmount')


# In[102]:


USAhousing.boxplot(column='LoanAmount',by='Gender')


# In[104]:


temp1 = USAhousing['Credit_History'].value_counts(ascending=True)
temp2 = USAhousing.pivot_table(values='Gender',index=['Credit_History'])
print ('Frequency Table for Credit History:') 
print (temp1)

print ('\nProbility of getting loan for each Credit History class:')
print (temp2)


# In[105]:


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Gender')
ax1.set_title("Gender by Credit_History")
temp1.plot(kind='bar')
ax2 = fig.add_subplot(122)
ax2.set_xlabel('Credit_History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title("Probability of getting loan by credit history")
temp2.plot(kind='bar')


# In[34]:


sns.distplot(USAhousing['LoanAmount'])


# In[48]:


USAhousing.corr()


# In[86]:


X=USAhousing [['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']]
y=USAhousing['CoapplicantIncome']


# In[87]:


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.4,random_state=101)


# In[88]:


from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X_train, y_train)


# In[89]:


predictions = lm.predict(X_test)


# In[90]:


plt.scatter(y_test,predictions,color = 'blue')
#plt.scatter(X_test, y_test, color = 'red')
plt.title('Prediksi Credit history Brdsr Gender(Training set)')
plt.xlabel('Credit_History')
plt.ylabel('Gender')
plt.show()
#plt.scatter(y_train,predictions,color = 'blue')
#plt.scatter(X_test, y_test, color = 'red')
#plt.title('Prediksi Credit history Brdsr Gender')
#plt.xlabel('Credit_History')
#plt.ylabel('Gender')
#plt.show()


# In[106]:


print(predictions)


# In[111]:


from sklearn.metrics import mean_squared_error, r2_score
error = y_test-predictions
print('Coefficients: \n', lm.coef_)
print("Mean squared error: %.2f"% mean_squared_error(y_test, predictions))
print(r2_score(y_test,predictions))

