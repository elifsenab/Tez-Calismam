#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import KFold
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


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lasso_model = Lasso(random_state=42).fit(X_train, y_train)
y_pred = lasso_model.predict(X_test)


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test,y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("RMSE: %.4f" % rmse)
print("MAPE: %.4f" % mape)
print("MAE: %.4f" % mae)


# In[17]:


k_fold = KFold(n_splits=7)
lamdalar = np.random.uniform(0,1,1000)
lassocv = LassoCV(alphas = lamdalar, max_iter = 7000, cv = k_fold)
lassocv.fit(X_train, y_train)


# In[18]:


lassocv.alpha_


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
lasso_model = Lasso(lassocv.alpha_, random_state=42).fit(X_train, y_train)

y_pred = lasso_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test,y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("RMSE: %.4f" % rmse)
print("MAPE: %.4f" % mape)
print("MAE: %.4f" % mae)


# In[ ]:




