#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
df_yield = pd.read_csv('C:/course/541/crop-yield-prediction-dataset/yield.csv')
df_yield.shape
df_yield.head()
df_yield.tail(10)


# In[2]:


df_yield = df_yield.rename(index=str, columns={"Value": "hg/ha_yield"})
df_yield.head()
# drop unwanted columns.
df_yield = df_yield.drop(['Year Code','Element Code','Element','Year Code','Area Code','Domain Code','Domain','Unit','Item Code'], axis=1)
df_yield.head()
df_yield.describe()
df_yield.info()


# In[3]:


df_rain = pd.read_csv('C:/course/541/crop-yield-prediction-dataset/rainfall.csv')
df_rain.head()


# In[4]:


df_rain = pd.read_csv('C:/course/541/crop-yield-prediction-dataset/rainfall.csv')
df_rain.head()
df_rain.tail()
df_rain = df_rain.rename(index=str, columns={" Area": 'Area'})
df_rain.info()


# In[5]:


df_rain['average_rain_fall_mm_per_year'] = pd.to_numeric(df_rain['average_rain_fall_mm_per_year'],errors = 'coerce')
df_rain.info()


# In[6]:


df_rain = df_rain.dropna()
df_rain.describe()


# In[7]:


yield_df = pd.merge(df_yield, df_rain, on=['Year','Area'])
yield_df.head()


# In[8]:


yield_df.describe()


# In[9]:


df_pes = pd.read_csv('C:/course/541/crop-yield-prediction-dataset/pesticides.csv')
df_pes.head()


# In[10]:


df_pes = df_pes.rename(index=str, columns={"Value": "pesticides_tonnes"})
df_pes = df_pes.drop(['Element','Domain','Unit','Item'], axis=1)
df_pes.head()


# In[11]:


df_pes.describe()


# In[12]:


df_pes.info()


# In[13]:


yield_df = pd.merge(yield_df, df_pes, on=['Year','Area'])
yield_df.shape
(18949, 6)
yield_df.head()


# In[14]:


avg_temp=  pd.read_csv('C:/course/541/crop-yield-prediction-dataset/temp.csv')
avg_temp.head()


# In[15]:


avg_temp.describe()


# In[16]:


avg_temp = avg_temp.rename(index=str, columns={"year": "Year", "country":'Area'})
avg_temp.head()


# In[17]:


yield_df = pd.merge(yield_df,avg_temp, on=['Area','Year'])
yield_df.head()


# In[18]:


yield_df.shape


# In[19]:


yield_df.describe()


# In[20]:


yield_df.isnull().sum()


# In[21]:


yield_df['Area'].nunique()


# In[22]:


yield_df.groupby(['Area'],sort=True)['hg/ha_yield'].sum().nlargest(10)


# In[23]:


yield_df.groupby(['Item','Area'],sort=True)['hg/ha_yield'].sum().nlargest(10)


# In[24]:


yield_df.head()


# In[25]:


yield_df_onehot = pd.get_dummies(yield_df, columns=['Area', 'Item'], prefix=['Country', 'Item'], dtype=int)
features = yield_df_onehot.drop('hg/ha_yield', axis=1)
label = yield_df['hg/ha_yield']
features.head()


# In[26]:


features = features.drop(['Year'], axis=1)
features.info()


# In[27]:


features.head()


# In[28]:


scaler=MinMaxScaler()
features=scaler.fit_transform(features) 
features


# In[29]:


train_data, test_data, train_labels, test_labels = train_test_split(features, label, test_size=0.2, random_state=42)


# In[30]:


yield_df_onehot = yield_df_onehot.drop(['Year'], axis=1)


# In[31]:


yield_df_onehot.head()


# In[32]:


test_df=pd.DataFrame(test_data,columns=yield_df_onehot.loc[:, yield_df_onehot.columns != 'hg/ha_yield'].columns)
cntry=test_df[[col for col in test_df.columns if 'Country' in col]].stack()[test_df[[col for col in test_df.columns if 'Country' in col]].stack()>0]
cntrylist=list(pd.DataFrame(cntry).index.get_level_values(1))
countries=[i.split("_")[1] for i in cntrylist]
itm=test_df[[col for col in test_df.columns if 'Item' in col]].stack()[test_df[[col for col in test_df.columns if 'Item' in col]].stack()>0]
itmlist=list(pd.DataFrame(itm).index.get_level_values(1))
items=[i.split("_")[1] for i in itmlist]


# In[33]:


test_df.head()


# In[34]:


test_df.drop([col for col in test_df.columns if 'Item' in col],axis=1,inplace=True)
test_df.drop([col for col in test_df.columns if 'Country' in col],axis=1,inplace=True)
test_df.head()


# In[35]:


test_df['Country']=countries
test_df['Item']=items
test_df.head()


# In[36]:


clf=linear_model.LinearRegression()
model=clf.fit(train_data,train_labels)

test_df["yield_predicted"]= model.predict(test_data)
test_df["yield_actual"]=pd.DataFrame(test_labels)["hg/ha_yield"].tolist()
test_group=test_df.groupby("Item")


# In[37]:


fig, ax = plt.subplots() 

ax.scatter(test_df["yield_actual"], test_df["yield_predicted"],edgecolors=(0, 0, 0))

ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title("Actual vs Predicted")
plt.show()


# In[38]:


# Assuming test_df["yield_actual"] contains the actual values and test_df["yield_predicted"] contains the predicted values
r2 = r2_score(test_df["yield_actual"], test_df["yield_predicted"])

print(f"R-squared: {r2}")


# In[39]:


# Calculate Mean Absolute Error
mae = mean_absolute_error(test_df["yield_actual"], test_df["yield_predicted"])
print("Mean Absolute Error:", mae)


# In[44]:


# Calculate Mean Squared Error
mse = mean_squared_error(test_df["yield_actual"], test_df["yield_predicted"])
print("Mean Squared Error:", mse)


# In[45]:


clf=RandomForestRegressor()
model=clf.fit(train_data,train_labels)

test_df["yield_predicted"]= model.predict(test_data)
test_df["yield_actual"]=pd.DataFrame(test_labels)["hg/ha_yield"].tolist()
test_group=test_df.groupby("Item")


# In[46]:


fig, ax = plt.subplots() 

ax.scatter(test_df["yield_actual"], test_df["yield_predicted"],edgecolors=(0, 0, 0))

ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title("Actual vs Predicted")
plt.show()


# In[47]:


# Assuming test_df["yield_actual"] contains the actual values and test_df["yield_predicted"] contains the predicted values
r2 = r2_score(test_df["yield_actual"], test_df["yield_predicted"])

print(f"R-squared: {r2}")


# In[48]:


# Calculate Mean Absolute Error
mae = mean_absolute_error(test_df["yield_actual"], test_df["yield_predicted"])
print("Mean Absolute Error:", mae)


# In[49]:


# Calculate Mean Squared Error
mse = mean_squared_error(test_df["yield_actual"], test_df["yield_predicted"])
print("Mean Squared Error:", mse)


# In[50]:


clf=XGBRegressor()
model=clf.fit(train_data,train_labels)

test_df["yield_predicted"]= model.predict(test_data)
test_df["yield_actual"]=pd.DataFrame(test_labels)["hg/ha_yield"].tolist()
test_group=test_df.groupby("Item")


# In[51]:


fig, ax = plt.subplots() 

ax.scatter(test_df["yield_actual"], test_df["yield_predicted"],edgecolors=(0, 0, 0))

ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title("Actual vs Predicted")
plt.show()


# In[52]:


# Assuming test_df["yield_actual"] contains the actual values and test_df["yield_predicted"] contains the predicted values
r2 = r2_score(test_df["yield_actual"], test_df["yield_predicted"])

print(f"R-squared: {r2}")


# In[53]:


# Calculate Mean Squared Error
mse = mean_squared_error(test_df["yield_actual"], test_df["yield_predicted"])
print("Mean Squared Error:", mse)


# In[54]:


# Calculate Mean Absolute Error
mae = mean_absolute_error(test_df["yield_actual"], test_df["yield_predicted"])
print("Mean Absolute Error:", mae)

