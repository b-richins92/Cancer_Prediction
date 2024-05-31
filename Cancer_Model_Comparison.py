#!/usr/bin/env python
# coding: utf-8

# In[26]:


import os
import requests
from zipfile import ZipFile
from urllib.request import urlretrieve
import pandas as pd
import numpy as np
#import altair as alt
#from vega_datasets import data
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import statsmodels.api as sm
from sklearn.feature_selection import RFECV
from sklearn.ensemble import *
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import *
from sklearn.metrics import *
import pycaret
from pycaret.classification import *
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance

#pd.set_option('display.max_columns', None)


# In[28]:


df = pd.read_csv(r'C:\Users\tegan\Downloads\clean_df.csv')
df['BMI'] = 703 * df['Weight in Pounds'] / (df['Height in Inches'] ** 2)
df.drop(columns=['Weight in Pounds', 'Height in Inches',
       'General Health_Excellent', 'General Health_Fair',
       'General Health_Good', 'General Health_Poor',
       'General Health_Very good', 'CT Scan', 'CT for Cancer'], inplace=True)
df.rename(columns={'Ethnicity_Multiracial, non-Hispanic': 'Ethnicity_Multiracial non-Hispanic'}, inplace=True)


# In[3]:


df.sample(10)


# In[4]:


correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=False)


# In[5]:


df.columns


# In[7]:


# Separate predictors and target
X, y = df.drop('Cancer', axis=1), df.Cancer

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale data
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


# In[8]:


forest = RandomForestClassifier(n_estimators=100,
                                max_depth=10,
                                random_state=42)

forest.fit(X_train_std, y_train)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]


# In[9]:


plt.ylabel('Feature importance')
plt.bar(range(X_train.shape[1]), 
        importances[indices],
        align='center')

feat_labels = X.columns
plt.xticks(range(X_train.shape[1]), 
           feat_labels[indices], rotation=90)

plt.xlim([-1, 15])

plt.tight_layout()
plt.show()


# ## Compare models using original data (non-SMOTE)

# In[10]:


clf = setup(df, target='Cancer', session_id=123)

best_model = compare_models()


# ## Compare models using SMOTE data

# In[11]:


smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
train_resampled = pd.concat([pd.DataFrame(X_train_resampled, columns=X_train.columns), pd.Series(y_train_resampled, name='Cancer')], axis=1)


# In[12]:


clf = setup(train_resampled, target='Cancer', session_id=125)

best_model = compare_models()


# ## KNN Model (SMOTE data)

# In[13]:


clf_setup = setup(data=train_resampled, target='Cancer', session_id=42)

knn_model = create_model('knn')

final_model = finalize_model(knn_model)

predictions = predict_model(final_model, data=X_test)


# ## KNN recall on test data

# In[14]:


y_true = y_test
y_pred = predictions['prediction_label']  # 'Label' column contains the predicted labels by PyCaret

recall = recall_score(y_true, y_pred, average='binary')  # Use 'binary' for binary classification
print(f"Recall Score: {recall}")


# ## Random Forest Model (SMOTE data)

# In[15]:


clf_setup = setup(data=train_resampled, target='Cancer', session_id=42)

rf_model = create_model('rf')

final_model = finalize_model(rf_model)

predictions = predict_model(final_model, data=X_test)


# In[28]:


print(rf_model)


# In[6]:


# Separate predictors and target
X, y = df.drop('Cancer', axis=1), df.Cancer

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale data
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


# In[8]:


smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_std, y_train)


# In[9]:


forest = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='sqrt',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_samples_leaf=1,
                       min_samples_split=2, min_weight_fraction_leaf=0.0,
                       monotonic_cst=None, n_estimators=100, n_jobs=-1,
                       oob_score=False, random_state=42, verbose=0,
                       warm_start=False)

forest.fit(X_train_resampled, y_train_resampled)


# In[16]:


y_pred = forest.predict(X_test_std)
recall = recall_score(y_test, y_pred)
recall


# In[18]:


y_pred = forest.predict(X_train_std)
recall = recall_score(y_train, y_pred)
recall


# In[ ]:


'''# Separate predictors and target
X, y = df.drop('Cancer', axis=1), df.Cancer

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale data
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_std, y_train)

forest = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='sqrt',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_samples_leaf=1,
                       min_samples_split=2, min_weight_fraction_leaf=0.0,
                       monotonic_cst=None, n_estimators=100, n_jobs=-1,
                       oob_score=False, random_state=42, verbose=0,
                       warm_start=False)

forest.fit(X_train_resampled, y_train_resampled)

y_test_pred = forest.predict(X_test_std)
test_recall = recall_score(y_test_pred, y_pred)
# test_recall = 0.046

y_train_pred = forest.predict(X_train_std)
train_recall = recall_score(y_train, y_train_pred)
# train_recall = 0.996'''


# In[ ]:


forest = RandomForestClassifier(random_state=42, n_jobs=-1)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [5, 15, 30]
}

grid_search = GridSearchCV(estimator=forest, param_grid=param_grid, cv=5, scoring='recall', n_jobs=-1)
grid_search.fit(X_train_resampled, y_train_resampled)

# Use the best model found
best_forest = grid_search.best_estimator_

# Evaluate on training data
y_train_pred = best_forest.predict(X_train_std)
train_recall = recall_score(y_train, y_train_pred)
print(f'Train Recall: {train_recall}')

# Evaluate on test data
y_test_pred = best_forest.predict(X_test_std)
test_recall = recall_score(y_test, y_test_pred)
print(f'Test Recall: {test_recall}')


# In[6]:


train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)


# In[7]:


clf = setup(train_df, target='Cancer', fix_imbalance=True)

best_model = compare_models()


# In[10]:


nb_model = create_model('nb')


# In[17]:


X, y = test_df.drop('Cancer', axis=1), test_df.Cancer


# In[18]:


y_pred = nb_model.predict(X)
recall = recall_score(y, y_pred)


# In[19]:


recall


# In[30]:


features = df.columns.to_list()
features.remove('Cancer')


# In[31]:


# # Calculate permutation importance
# perm_importance = permutation_importance(nb_model, df[features], df['Cancer'], n_repeats=30, random_state=42)


# # Plot permutation importance
# import matplotlib.pyplot as plt

# sorted_idx = perm_importance.importances_mean.argsort()
# plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
# plt.xlabel("Permutation Importance")
# plt.title("Permutation Importance (Naive Bayes Model)")
# plt.show()


# In[ ]:




