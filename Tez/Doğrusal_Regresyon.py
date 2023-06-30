#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
from scipy.stats import kstest, norm, skew, stats
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[3]:


data = pd.read_excel("C:/Users/elifs/Desktop/EvFiyatVeriSeti.xlsx")
df = data.copy()


# In[4]:


fiyat = df["Fiyat"]
Donusum = np.log(fiyat)
df["Fiyat"] = Donusum


# In[5]:


oda_sayisi = df["Oda_Sayısı"]
# Kukla değişken dönüşümü
kukla_degiskenler = pd.get_dummies(oda_sayisi, prefix="Oda_Sayısı_Kukla", drop_first=True)
df = pd.concat([df, kukla_degiskenler], axis=1)
df = df.drop("Oda_Sayısı", axis=1)


# In[6]:


bina_yasi = df["Bina_Yaşı"]
kukla_degiskenler = pd.get_dummies(bina_yasi, prefix="Bina_Yaşı_Kukla", drop_first=True)
df = pd.concat([df, kukla_degiskenler], axis=1)
df = df.drop("Bina_Yaşı", axis=1)


# In[7]:


mahalle = df["Mahalle"]
kukla_degiskenler = pd.get_dummies(mahalle, prefix="Mahalle_Kukla", drop_first=True)
df = pd.concat([df, kukla_degiskenler], axis=1)
df = df.drop("Mahalle", axis=1)


# In[8]:


y = df['Fiyat']
X = df.drop("Fiyat", axis=1)


# In[9]:


X = sm.add_constant(X)
# Logaritmik doğrusal model
model = sm.OLS(y, X).fit()
print(model.summary())


# In[12]:


# VIF değerleri
vif = pd.DataFrame()
vif["Değişken"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif)


# In[14]:


y_pred = model.predict(X)

# MAPE hesaplama
percentage_errors = np.abs((y - y_pred) / y)
mape = np.mean(percentage_errors) * 100

print("MAPE:", mape)


# In[15]:


def calculate_mape(y, y_pred):
    if len(y) != len(y_pred):
        raise ValueError("The lengths of actual_values and predicted_values should be equal.")

    percentage_errors = np.abs((np.array(y) - np.array(y_pred)) / np.array(y))
    mape = np.mean(percentage_errors) * 100

    return mape


# In[13]:


y_pred = model.predict(X)


# In[21]:


rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)
mape = mean_absolute_percentage_error(y,y_pred)

print("RMSE: %.4f" % rmse)
print("MAE: %.4f" % mae)
print("MAPE: %.4f" % mape)


# In[ ]:




