import numpy as np
import pandas as pd
import re
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, f1_score, recall_score,
        classification_report, accuracy_score, roc_auc_score, precision_score)

superhero = pd.read_csv("superheroes.csv")

#Num of superheroes
superhero.info()

superhero['Alignment'].head()

#Num of good superheroes
superhero[superhero['Alignment'] == 1].count()

#Num of bad superheroes
superhero[superhero['Alignment'] == 0].count()

#Average number of powers.
num_powers = superhero.sum(axis = 1)

num_powers.mean()

#Average number of good powers
num_good_pow = superhero[superhero['Alignment'] == 1].sum(axis = 1)

num_good_pow.mean()

#Average number of bad powers
num_bad_pow = superhero[superhero['Alignment'] == 0].sum(axis = 1)

num_bad_pow.mean()

#Name of superhero with most powers
superhero.sum(axis = 1).max()

superhero[superhero.sum(axis = 1) == 49]

#Most common superpower
superhero.sum().drop(['name','Alignment']).max()

#Train Test Split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

superhero = superhero[[col for col in superhero if col not in ['Alignment']]
                            + ['Alignment']]

features = superhero.columns[1:-1]
target = superhero.columns[-1]

X = superhero[features]
y = superhero[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                stratify = superhero.Alignment,
                                                random_state = 1225)

nb = MultinomialNB()
nb = nb.fit(X_train, y_train)

y_pred_train = nb.predict(X_train)
y_pred = nb.predict(X_test)
y_prob_train = nb.predict_proba(X_train)[:,1]
y_prob = nb.predict_proba(X_test)

confusion_matrix(y_train, y_pred_train)
accuracy_score(y_train, y_pred_train)
precision_score(y_train, y_pred_train)
recall_score(y_train, y_pred_train)
f1_score(y_train, y_pred_train)
roc_auc_score(y_train, y_prob_train)

#specificity
confusion_matrix(y_train, y_pred_train)

total_no = 52+85
print(52/total_no)

#Cross Validation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

lasso = LogisticRegression(penalty = 'l1', C = 0.4, solver = 'liblinear')
lasso = lasso.fit(X, y) 
lasso_score = cross_val_score(lasso, X, y, scoring='roc_auc', cv = 10)

np.mean(lasso_score)
np.std(lasso_score)

nb = MultinomialNB()
nb = nb.fit(X, y)

nb_score = cross_val_score(nb, X, y, scoring='roc_auc', cv = 10)

np.mean(nb_score)
np.std(nb_score)

#Lasso with all data
lasso = LogisticRegression(penalty = 'l1', C = 0.4, solver = 'liblinear')
lasso = lasso.fit(X_train, y_train)

#coef_


coef_dict = {}
for coef, feat in zip(model_1.coef_[0,:],model_1_features):
    coef_dict[feat] = coef

coef_dict

#future superheros
future_superhero = pd.read_csv("superheroes_future.csv")

lasso.predict_proba(future_superhero)[:,1]

#Clustering
#remove strength
superhero = superhero.drop('Super Strength', axis = 1)

from sklearn.cluster import KMeans

wcss = []
for k in range(1,10):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(np.arange(1,10),wcss,marker="o")
plt.xlabel('Number of Clusters, K')
plt.ylabel('WCSS')

k_centers = 2
km = KMeans(k_centers)
km.fit(X)
wcss = km.inertia_

order_centroids = km.cluster_centers_.argsort()[:,::-1] 
for i in range(k_centers): 
    print("Cluster %d:" % i, end='') 
    for ind in order_centroids[i,:10]: 
        print(' %s' % features, end='') 
    print()





