#!/usr/bin/env python
# coding: utf-8

# In[183]:


#importation des librairies 
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


# In[184]:


#téléchargement des données training utilisés dans la régression
file_name1= "/Users/IGaraouch/OneDrive - EY/Desktop/auto-insurance/train_auto.csv"
df_train = pd.read_csv(file_name1, sep=";")
df_train


# In[185]:


#téléchargement et affichage de la base de données qui contient les variables qu'on va utiliser pour le test
file_name2= "/Users/IGaraouch/OneDrive - EY/Desktop/auto-insurance/test_auto.csv" 
df_test = pd.read_csv(file_name2, sep=";")
df_test


# In[186]:


#affichage des 5 premières lignes de la base 
df_train.head()


# In[187]:


#taille de la base training
df_train.shape


# In[188]:


#différentes variables dans la base training qui peuvent potentiellemnt etre incluses dans la régression pour prédire la variable target
print(list(df_train.columns))
print(list(df_test.columns))


# In[189]:


#une première description des variables de la base de training
df_train.info()


# In[190]:


#caractéristiques des différentes colonnes de la base training
df_train.describe()


# In[191]:


#les valeurs nulles dans chaque colonne 
df_train.isnull().sum()


# 
# 
# ici on voit que les variables "YOJ", "INCOME", "JOB", "CAR_AGE" contiennent le plus de valeurs nulles qu'on va chercher à supprimer de la base:
# 
# 
# 1- étude des corrélations entre les potentielles variables explicatives et la variables target: On rentient celles qui sont le plus corrélées à notre variables target
# 
# 
# 2- Parmi celles-ci, voir s'il y en a qui contiennent des valeurs nulles et s'il y en a qui ne sont pas de type numérique et faire des modifications si nécessaires
# 

# In[192]:


matrice_corr = df_train.corr().round(1)
sns.heatmap(data=matrice_corr, annot=True)


# On remarque une corrélation nulle ou trés faible  entre la variable target ( TARGET_FLAG ) et les variables INDEX, KIDSRIV, AGE, HOMEKIDS, YOJ, TRAVTIME,TIF,CAR_AGE
# on va supprimer ses varibles de nos données de trainings

# In[193]:


df_train.drop(['INDEX', 'KIDSDRIV', 'AGE', 'HOMEKIDS', 'YOJ', 'TRAVTIME','TIF','CAR_AGE'], axis=1, inplace=True)
df_test.drop(['INDEX', 'KIDSDRIV', 'AGE', 'HOMEKIDS', 'YOJ', 'TRAVTIME','TIF','CAR_AGE'], axis=1, inplace=True)


# aprés avor supprimer les variables les moins corrélées à notre variable target TARGET_FLAG on va refaire une étude des corrélations avec les autres variables restantes pour en rentenir les plus pertinentes à la fin

# In[194]:


matrice_corr = df_train.corr().round(1)
sns.heatmap(data=matrice_corr, annot=True)


# Parmi les variables numériques celles qui sont le plus corrélés à TARGET_FLAG sont TARGET_AMT, CLM_FREQ et MVR_PTS
# Ainsi on retient ces trois variables explicatives pour faire la régression linéaire

# In[195]:


fig, axes=plt.subplots(1, 3, figsize=(15, 4))
sns.regplot(ax=axes[0], x='TARGET_AMT', y='TARGET_FLAG', data=df_train, scatter_kws={'s':10})
sns.regplot(ax=axes[1], x='CLM_FREQ', y='TARGET_FLAG', data=df_train, scatter_kws={'s':10})
sns.regplot(ax=axes[2], x='MVR_PTS', y='TARGET_FLAG', data=df_train, scatter_kws={'s':10})
fig.suptitle('explanatory variable vs target_flag', fontsize=16)


# J'ai refait ici un scatter plot qui permet encore une fois de visualiser la corrélation entre notre variable target et le variables explicatives: On voit que TARGET_AMT est la plus corrélée à TARGET_FLAG ( meme si cette corrélation rest pas trop élevée mais c'est la plus importante).
# Les variables CLM_FREQ et MVR_PTS sont corrélés positivement à TARGET_FLAG mais la corrélation reste faible.

# In[196]:


#on utilise seulement 3 variables explicatives:TARGET_AMT, CLM_FREQ et MVR_PTS
X=pd.DataFrame(np.c_[df_train['TARGET_AMT'],df_train['CLM_FREQ'],df_train['MVR_PTS']], columns = ['TARGET_AMT','MVR_PTS','CLM_FREQ'])
Y= df_train['TARGET_FLAG']

#base d'apprentissage et base de test
from sklearn.model_selection import train_test_split
 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[197]:


#entrainement du modèle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
 
lmodellineaire = LinearRegression()
lmodellineaire.fit(X_train, Y_train)


# In[198]:


# Evaluation du training set
from sklearn.metrics import r2_score
y_train_predict = lmodellineaire.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)

print('La performance du modèle sur la base dapprentissage')
print('--------------------------------------')
print('Lerreur quadratique moyenne est {}'.format(rmse))
print('le score R2 est {}'.format(r2))
print('\n')
 
# model evaluation for testing set
y_test_predict = lmodellineaire.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2 = r2_score(Y_test, y_test_predict)

print('La performance du modèle sur la base de test')
print('--------------------------------------')
print('Lerreur quadratique moyenne est {}'.format(rmse))
print('le score R2 est {}'.format(r2))


# In[199]:


print(y_train_predict)
df_y_train_predict=pd.DataFrame(y_train_predict)
# enregistrement du dataframe des prédictions sous le fomat d'un fichier csv
prediction_csv_data = df_y_train_predict.to_csv('prediction.csv', index = True) 
print('\nCSV String:\n', prediction_csv_data) 

