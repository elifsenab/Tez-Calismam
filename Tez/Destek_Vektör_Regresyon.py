#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


data = pd.read_excel("C:/Users/elifs/Desktop/EvFiyatVeriSeti.xlsx")
df = data.copy()


# In[4]:


fiyat = df["Fiyat"]
Donusum = np.log(fiyat)
df["Fiyat"] = Donusum


# In[5]:


oda_sayisi = df["Oda_Sayısı"]
kukla_degiskenlerrr = pd.get_dummies(oda_sayisi, prefix="Oda_Sayısı_Kukla", drop_first=True)
df = pd.concat([df, kukla_degiskenlerrr], axis=1)
df = df.drop("Oda_Sayısı", axis=1)


# In[6]:


bina_yasi = df["Bina_Yaşı"]
kukla_degiskenlerr = pd.get_dummies(bina_yasi, prefix="Bina_Yaşı_Kukla", drop_first=True)
df = pd.concat([df, kukla_degiskenlerr], axis=1)
df = df.drop("Bina_Yaşı", axis=1)


# In[7]:


mahalle = df["Mahalle"]
kukla_degiskenler = pd.get_dummies(mahalle, prefix="Mahalle_Kukla", drop_first=True)
df = pd.concat([df, kukla_degiskenler], axis=1)
df = df.drop("Mahalle", axis=1)


# In[8]:


y = df['Fiyat']
X = df.drop("Fiyat", axis=1)


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svr_model = SVR()
svr_model.fit(X_train, y_train)
y_pred = svr_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test,y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("RMSE: %.4f" % rmse)
print("MAPE: %.4f" % mape)
print("MAE: %.4f" % mae)


# ### Model Tuning

# In[46]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[22]:


from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svr_model = SVR(kernel='linear')
param_grid = {
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.5, 1]}
grid_search = GridSearchCV(svr_model, param_grid, cv=5).fit(X_train, y_train)
grid_search.best_params_
y_pred = grid_search.predict(X_test)


# In[23]:


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test,y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("RMSE: %.4f" % rmse)
print("MAPE: %.4f" % mape)
print("MAE: %.4f" % mae)


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svr_model = SVR(kernel='rbf')
param_grid = {
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.5, 1]}
grid_search = GridSearchCV(svr_model, param_grid, cv=5).fit(X_train, y_train)
grid_search.best_params_
y_pred = grid_search.predict(X_test)


# In[16]:


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test,y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("RMSE: %.4f" % rmse)
print("MAPE: %.4f" % mape)
print("MAE: %.4f" % mae)


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svr_model = SVR(kernel='poly')
param_grid = {
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.5, 1]}
grid_search = GridSearchCV(svr_model, param_grid, cv=5).fit(X_train, y_train)
grid_search.best_params_
y_pred = grid_search.predict(X_test)


# In[18]:


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test,y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("RMSE: %.4f" % rmse)
print("MAPE: %.4f" % mape)
print("MAE: %.4f" % mae)


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svr_model = SVR(kernel='sigmoid')
param_grid = {
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.5, 1]}
grid_search = GridSearchCV(svr_model, param_grid, cv=5).fit(X_train, y_train)
grid_search.best_params_
y_pred = grid_search.predict(X_test)


# In[20]:


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test,y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("RMSE: %.4f" % rmse)
print("MAPE: %.4f" % mape)
print("MAE: %.4f" % mae)


# In[ ]:




