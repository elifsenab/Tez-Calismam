#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import pandas as pd
import numpy as np
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


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[21]:


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test,y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("RMSE: %.4f" % rmse)
print("MAPE: %.4f" % mape)
print("MAE: %.4f" % mae)


# ## Tuning

# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[35]:


cart_params = {"max_depth": [2,7,41,55,98,101],
              "min_samples_split": [3,5,3,46,35,142]}


# In[22]:


cart_model = DecisionTreeRegressor(random_state=42)


# In[37]:


cart_cv_model = GridSearchCV(cart_model, cart_params, cv = 5).fit(X_train, y_train)


# In[38]:


cart_cv_model.best_params_


# In[18]:


cart_model = DecisionTreeRegressor(max_depth = 7, min_samples_split = 3).fit(X_train, y_train)


# In[23]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
cart_params = {
    "max_depth": [10, 15, 20, 25, 30],
    "min_samples_split": [2, 3, 4, 5],
    "min_samples_leaf": [1, 2, 3, 4, 5]
}
cart_model = DecisionTreeRegressor(random_state=42)
cart_cv_model = GridSearchCV(cart_model, cart_params, cv = 5).fit(X_train, y_train)
cart_cv_model.best_params_
cart_model = DecisionTreeRegressor(max_depth = 7, min_samples_split = 3).fit(X_train, y_train)


# In[24]:


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test,y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("RMSE: %.4f" % rmse)
print("MAPE: %.4f" % mape)
print("MAE: %.4f" % mae)


# In[ ]:




