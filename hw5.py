import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
recall_score, roc_auc_score, roc_curve, accuracy_score)

#1. Import the data. Note that the dataset is tab delimited

movies = pd.read_csv('moviereviews.tsv', sep = '\t')

movies.head(5)


#2. Check for and remove missing values and blank strings (if any)

print(movies.info())

print(movies[movies.review.isnull() == True])

movies.dropna(inplace = True)

print(movies.info())

#3. Split the data into a training set and a test set. Use test_size=0.4, and
#   random_state=713

train, test = train_test_split(movies, test_size=.4, stratify=movies.label, random_state=713)

print(train.shape)
print(test.shape)

#4. Vectorize the data using TF-IDF. To keep the X matrix from being too large,
#   use min_df = 50 and remove stop words (see below). Be sure that all model
#   development is with the training data (that is, fit the TD-IDF transformer on
#   the training data, then transform to both training and test data).

#stopwords from nltk.corpus.stopwords.words('english')
sw = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
        "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
        'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
        'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
        'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
        'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
        'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
        'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
        'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
        'about', 'against', 'between', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
        'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
        'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
        'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
        'not', 'only', 'own', 'same', 'so', 'than', 'too', 'ver', 's', 't',
        'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now',
        'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",
        'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',
        "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma',
        'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan',
        "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",
        'won', "won't", 'wouldn', "wouldn't"]

tfidf = TfidfVectorizer(min_df = 50, stop_words = sw)
tfidf.fit(train['review'])
X_train = tfidf.transform(train['review'])
X_test = tfidf.transform(test['review'])

y_train = (train['label'] == 'neg').astype(int)
y_test = (test['label'] == 'neg').astype(int)



#5. Build classifiers: using the default tuning parameter values, fit models
#   using:

#      * Multinomial Naive Bayes 
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
nb.fit(X_train, y_train)
yhat = nb.predict(X_test)
y_prob = nb.predict_proba(X_test)[:,1]

print(roc_auc_score(y_test, y_prob))
print(accuracy_score(y_test, yhat))
print(f1_score(y_test, yhat))


#      * Decision Tree 
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
yhatdt = (dt.predict(X_test))
y_probdt = dt.predict_proba(X_test)[:,1]

print(roc_auc_score(y_test, y_probdt))
print(accuracy_score(y_test, yhatdt))
print(f1_score(y_test, yhatdt))

#      * Random Forest 
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_depth = 100, n_estimators = 500, n_jobs = -1)
rf.fit(X_train, y_train)
yhatrf = (rf.predict(X_test))
y_probrf = rf.predict_proba(X_test)[:,1]

print(roc_auc_score(y_test, y_probrf))
print(accuracy_score(y_test, yhatrf))
print(f1_score(y_test, yhatrf))

#      * Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(max_depth = 100, n_estimators = 500)
gb.fit(X_train, y_train)
yhatgb = (gb.predict(X_test))
y_probgb = gb.predict_proba(X_test)[:,1]

print(roc_auc_score(y_test, y_probgb))
print(accuracy_score(y_test, yhatgb))
print(f1_score(y_test, yhatgb))


#6. Report the training and testing accuracy, F1 score, and AUC for each model
#   in the previous problem. What model performs the best?

# Naive Bayes F1: 0.8898233809924306
# Naive Bayes Accuracy: 0.8904682274247492
# Naive Bayes AUC: 0.9566937170725159

# Decision Tree F1: 0.740281224152192
# Decision Tree Accuracy: 0.7374581939799331
# Decision Tree AUC: 0.7374581939799331

# Random Forest F1: 0.8780075981426761
# Random Forest Accuracy: 0.879180602006689
# Random Forest AUC: 0.9486313771658036

# Gradient Boost F1: 0.7571369466280514
# Gradient Boost Accuracy: 0.7545986622073578
# Gradient Boost AUC: 0.8527232637218823

# With my results it looks like Naive Bayes performs the best.

#7. What words are most important for distinguishing between positive and
#   negative reviews? Compare the important features for your top two models.

# Naive Bayes important features:
neg_class_prob_sorted = nb.feature_log_prob_[0, :].argsort()
pos_class_prob_sorted = nb.feature_log_prob_[1, :].argsort()

print(np.take(tfidf.get_feature_names(), neg_class_prob_sorted[-10:]))
print(np.take(tfidf.get_feature_names(), pos_class_prob_sorted[-10:]))

# Random Forest important features:

Importance = pd.DataFrame({'Importance':rf.feature_importances_*100},
                            index=tfidf.get_feature_names())
Importance = Importance.iloc[rf.feature_importances_ > 0,:]
Importance = Importance.sort_values('Importance', axis=0, ascending=False).head(20)
Importance.plot(kind='barh', color='r', )
plt.xlabel('Variable Importance')
plt.gca().legend_ = None

print(Importance)
