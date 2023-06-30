#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np


# In[32]:


data = pd.read_excel("C:/Users/elifs/Desktop/EvFiyatVeriSeti.xlsx")
df = data.copy()


# In[19]:


df.head()


# In[28]:


skewness = stats.skew(df["Fiyat"])
print("Çarpıklık katsayısı: %.2f" % skewness)
kurtosis = stats.kurtosis(df["Fiyat"])
print("Basıklık katsayısı: %.2f" % kurtosis)


# In[23]:


fiyat = df["Fiyat"]
Donusum = np.log(fiyat)
df["Fiyat"] = Donusum


# In[26]:


skewness = stats.skew(df["Fiyat"])
print("Çarpıklık katsayısı: %.2f" % skewness)
kurtosis = stats.kurtosis(df["Fiyat"])
print("Basıklık katsayısı: %.2f" % kurtosis)


# In[33]:


plt.figure(figsize=(9, 3))
plt.plot(df.index, df.Fiyat, 'o', markersize=4)
plt.show()


# In[17]:


# Değişkenlerin korelasyon matrisi
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()


# In[34]:


fiyat = fiyat = df["Fiyat"]
brut_metrekare =  df["m²(Brüt)"]

sns.regplot(x=brut_metrekare, y=fiyat , data=df)

plt.xlabel('Brüt m²')
plt.ylabel('Fiyat')

plt.show()


# In[27]:


fiyat = fiyat = df["Fiyat"]
bulundugu_kat =  df["Bulunduğu_Kat"]

sns.boxplot(x=bulundugu_kat, y=fiyat, data=df)

plt.xlabel('Bulunduğu Kat')
plt.ylabel('Fiyat')

plt.show()


# In[25]:


fiyat = fiyat = df["Fiyat"]
oda_sayısı =  df["Oda_Sayısı"]

sns.boxplot(x=oda_sayısı, y=fiyat, data=df)

plt.xlabel('Oda Sayısı')
plt.ylabel('Fiyat')

plt.show()

