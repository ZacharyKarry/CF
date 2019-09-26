# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 19:42:53 2019

@author: Zachary Karry
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

gen = pd.read_csv('results.csv')
gen['Sample Name'] = gen['Sample Name'].str.lower()

can = pd.read_csv('cannabis.csv')
can['Strain'] = can['Strain'].str.lower()
can['Strain'] = can['Strain'].str.replace('-', ' ')

mer = pd.merge(gen, can, left_on='Sample Name', right_on='Strain')

mer = mer[mer['Rating'] >= 4]
mer = mer[mer['Rating'] != 5]
mer['Rating'] = mer['Rating'].replace({4.0: 0,
   4.1: 0,
   4.2: 0,
   4.3: 1,
   4.4: 2,
   4.5: 2,
   4.6: 3,
   4.7: 3,
   4.8: 3,
   4.9: 3})

cols = ['Sample Name',
       'Caryophyllene Oxide',
       'Linalool', 'Terpinolene', 
       'alpha-Humulene', 'alpha-Pinene', 
       'beta-Caryophyllene', 'beta-Myrcene', 'beta-Pinene',
       'delta-Limonene', 'delta-9 THC-A',
       'delta-9 THC', 'CBD-A', 'CBD', 'Rating']

mer = mer[cols]

X_cols = ['Caryophyllene Oxide',
       'Linalool', 'Terpinolene', 
       'alpha-Humulene', 'alpha-Pinene', 
       'beta-Caryophyllene', 'beta-Myrcene', 'beta-Pinene',
       'delta-Limonene', 'delta-9 THC-A',
       'delta-9 THC', 'CBD-A', 'CBD']
y_cols = ['Rating']
mer.dropna(inplace=True)

X_train, X_test, y_train, y_test = train_test_split(mer[X_cols], mer[y_cols], random_state=42)

# Create a random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(n_jobs=2, random_state=42)

# Train the Classifier to take the training features and learn how they relate
# to the training y (the species)
clf.fit(X_train, y_train)

#pd.crosstab(clf.predict(X_test),y_test['Rating'])
#list(zip(X_train, clf.feature_importances_))