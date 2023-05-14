#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as pn
import seaborn as sns
import matplotlib.pyplot as plt
df_test=pd.read_csv('Desktop/blackfriday_test.csv')
df_test.head()


# In[2]:


df_train=pd.read_csv('Desktop/blackfriday_train.csv')
df_train.head()


# In[3]:


df=df_test.append(df_train)
df.head()


# In[4]:


df


# In[5]:


df.info()


# In[6]:


df.drop(['User_ID'],axis=1,inplace=True)
df.head()


# # maping and finding unique value for age

# In[7]:


df['Gender']=df['Gender'].map({'F':0, 'M':1})


# In[8]:


df['Age'].unique()


# In[9]:


df


# In[10]:


df['Age'].unique()


# In[11]:


df['Age']=df['Age'].map({'46-50':5, '26-35':3, '36-45':4, '18-25':2, '51-55':6, '55+':7, '0-17':1})


# In[12]:


#pd.get_dummies(['Age'],drop_first=True)
#df['Age']=df['Age'].map({'0-17':1,'18-25':2,'26-35':3,'36-45':4,'46-50':5,'51-55':6,'55+':7})


# In[13]:


df['Age']


# In[14]:



from sklearn import preprocessing
  
    # label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
  
# Encode labels in column 'species'.
df['Age']= label_encoder.fit_transform(df['Age'])
  
df['Age'].unique()


# In[15]:


df.head()


# # fixing categorical cities

# In[16]:


df_city=pd.get_dummies(['City_Category'],drop_first=True)
df_city.head()


# In[17]:


df.drop('City_Category',axis=1,inplace=True)
df.head()


# In[18]:


df.isnull().sum()


# In[19]:


df['Product_Category_2'].unique()


# In[20]:


df['Product_Category_2'].value_counts()


# In[21]:


df['Product_Category_2']=df['Product_Category_2'].fillna(df['Product_Category_2']).mode()[0]


# In[22]:


df['Product_Category_2'].isnull().sum()


# In[23]:


df['Product_Category_3'].unique()


# # replacing missing values mode

# In[24]:


df['Product_Category_2'].fillna(df['Product_Category_2'].mode()[0], inplace=True)
df[df['Product_Category_2'] == df['Product_Category_2'].mode()[0]]['Product_Category_2'].value_counts()


# In[25]:


df['Product_Category_2'].isnull().sum()


# # replacing Product_Category_3 missing values with mode

# In[26]:


df['Product_Category_3'].unique()


# In[27]:


df['Product_Category_3'].value_counts()


# In[28]:


df['Product_Category_3'].fillna(df['Product_Category_3'].mode()[0], inplace=True)
df[df['Product_Category_3'] == df['Product_Category_3'].mode()[0]]['Product_Category_3'].value_counts()


# In[29]:


df.head()


# In[30]:


df.shape


# In[31]:


df['Stay_In_Current_City_Years'].unique()


# In[32]:


df.head()


# In[33]:


df.info()


# # convert object into integer

# In[34]:


df['Stay_In_Current_City_Years'].unique()


# In[35]:


df.info()


# In[36]:


# Convert column to string
df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].astype(str)
# Replace '4+' with '4'
df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].str.replace('+', '')
# Convert column to integer
df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].astype(int)


# In[37]:


df.info()


# # visualization 

# In[38]:


sns.barplot('Age','Purchase', hue='Gender',data=df)


# # purchase of men is high than women

# In[39]:


#visualization purchase with occupation
sns.barplot('Occupation','Purchase', hue='Gender',data=df)


# In[40]:


sns.barplot('Product_Category_1','Purchase',hue='Gender',data=df)


# In[41]:


sns.barplot('Product_Category_2','Purchase',hue='Gender',data=df)


# In[42]:


df.head()


# # feature scaling

# In[43]:


df_test=df[df['Purchase'].isnull()]


# In[44]:


df_test=df[~df['Purchase'].isnull()]


# In[45]:


X=df_train.drop('Purchase',axis=1)


# In[46]:


X.head()


# In[47]:


X.shape


# In[48]:


Y=df_train['Purchase']


# In[49]:


Y


# In[50]:


Y.shape


# In[51]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.33, random_state=42)


# In[52]:


X_train.drop('Product_ID',axis=1,inplace=True)
X_test.drop('Product_ID',axis=1,inplace=True)


# In[53]:


from sklearn.preprocessing import StandardScaler
import numpy as np

# create dummy data
X_train = np.array([[1, 2], [3, 4], [5, 6]])
X_test = np.array([[7, 8], [9, 10]])

# initialize scaler
sc = StandardScaler()

# apply scaler to training data
X_train = sc.fit_transform(X_train)

# apply scaler to test data
X_test = sc.transform(X_test)

