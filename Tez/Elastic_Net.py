#!/usr/bin/env python
# coding: utf-8

# In[10]:


from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_excel("C:/Users/elifs/Desktop/EvFiyatVeriSeti.xlsx")
df = data.copy()


# In[3]:


fiyat = df["Fiyat"]
Donusum = np.log(fiyat)
df["Fiyat"] = Donusum


# In[4]:


oda_sayisi = df["Oda_Sayısı"]
kukla_degiskenlerrr = pd.get_dummies(oda_sayisi, prefix="Oda_Sayısı_Kukla", drop_first=True)
df = pd.concat([df, kukla_degiskenlerrr], axis=1)
df = df.drop("Oda_Sayısı", axis=1)


# In[5]:


bina_yasi = df["Bina_Yaşı"]
kukla_degiskenlerr = pd.get_dummies(bina_yasi, prefix="Bina_Yaşı_Kukla", drop_first=True)
df = pd.concat([df, kukla_degiskenlerr], axis=1)
df = df.drop("Bina_Yaşı", axis=1)


# In[6]:


mahalle = df["Mahalle"]
kukla_degiskenler = pd.get_dummies(mahalle, prefix="Mahalle_Kukla", drop_first=True)
df = pd.concat([df, kukla_degiskenler], axis=1)
df = df.drop("Mahalle", axis=1)


# In[7]:


y = df['Fiyat']
X = df.drop("Fiyat", axis=1)


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

enet_model = ElasticNet(random_state=42).fit(X_train, y_train)
y_pred = enet_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test,y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("RMSE: %.4f" % rmse)
print("MAPE: %.4f" % mape)
print("MAE: %.4f" % mae)


# In[24]:


param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
              'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}

enet_model = ElasticNet(random_state=42)
enet_model_cv = GridSearchCV(enet_model, param_grid, cv=7).fit(X_train, y_train)

alpha = enet_model_cv.best_params_['alpha']
l1_ratio = enet_model_cv.best_params_['l1_ratio']


# In[25]:


enet_reg = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

enet_reg.fit(X_train, y_train)

y_pred = enet_reg.predict(X_test)


# In[26]:


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test,y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("RMSE: %.4f" % rmse)
print("MAPE: %.4f" % mape)
print("MAE: %.4f" % mae)


# In[ ]:




