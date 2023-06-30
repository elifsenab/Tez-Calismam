#!/usr/bin/env python
# coding: utf-8

# In[54]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor


# In[78]:


data = pd.read_excel("C:/Users/elifs/Desktop/EvFiyatVeriSeti.xlsx")
df = data.copy()


# In[83]:


fiyat = df["Fiyat"]
Donusum = np.log(fiyat)
df["Fiyat"] = Donusum


# In[84]:


oda_sayisi = df["Oda_Sayısı"]
kukla_degiskenlerrr = pd.get_dummies(oda_sayisi, prefix="Oda_Sayısı_Kukla", drop_first=True)
df = pd.concat([df, kukla_degiskenlerrr], axis=1)
df = df.drop("Oda_Sayısı", axis=1)


# In[85]:


bina_yasi = df["Bina_Yaşı"]
kukla_degiskenlerr = pd.get_dummies(bina_yasi, prefix="Bina_Yaşı_Kukla", drop_first=True)
df = pd.concat([df, kukla_degiskenlerr], axis=1)
df = df.drop("Bina_Yaşı", axis=1)


# In[86]:


mahalle = df["Mahalle"]
kukla_degiskenler = pd.get_dummies(mahalle, prefix="Mahalle_Kukla", drop_first=True)
df = pd.concat([df, kukla_degiskenler], axis=1)
df = df.drop("Mahalle", axis=1)


# In[87]:


y = df['Fiyat']
X = df.drop("Fiyat", axis=1)


# In[95]:


df.head()


# In[129]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42).fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test,y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("RMSE: %.4f" % rmse)
print("MAPE: %.4f" % mape)
print("MAE: %.4f" % mae)


# ## Tuning

# In[133]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor().fit(X_train, y_train)


# In[134]:


params = {"max_depth": [5, 15, 25],
    "max_features": [4, 5, 9],
    "n_estimators": [100, 500, 800],
    "min_samples_split": [2, 3, 5]}


# In[135]:


cv_model = GridSearchCV(model, params, cv=5, n_jobs=-1).fit(X_train, y_train)


# In[136]:


cv_model.best_params_


# In[137]:


tuned = RandomForestRegressor(random_state = 42,
                             max_depth = 15,
                             max_features = 4,
                             min_samples_split = 5,
                             n_estimators = 500).fit(X_train, y_train)


# In[139]:


y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("RMSE: %.4f" % rmse)
print("MAPE: %.4f" % mape)
print("MAE: %.4f" % mae)


# ## Değişken Önem Düzeyi

# In[149]:


importance = model.feature_importances_
feature_names = X.columns

feature_names_updated = [ 'm²(Brüt)', 'Banyo_Sayısı', 'Bulunduğu_Kat', 'Balkon', 'Site_İçerisinde', '2+1',
                         '3+1', '4+1', '5-10 yaş', '11-15 yaş', '16-20 yaş',
                         '21-25 yaş', 'Cumhuriyet', 'Esenevler', 'İstiklal', 'Körfez',
                         'Mevlana', 'Mimarsinan', 'Yenimahalle', 'Yeşildere']

importance_df = pd.DataFrame({'Feature': feature_names_updated, 'Importance': importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.xlabel('Değişken Önem Düzeyi')
plt.ylabel("Değişken")
plt.show()


# In[ ]:




