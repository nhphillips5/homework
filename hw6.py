import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import re
import datetime



baseball = pd.read_csv("baseball.csv")

baseball = baseball.drop(columns = ['playerID', 'firstName', 'lastName'])
# 1.Scale the data with the MaxMinScaler from sklearn.preprocessing.
from sklearn.preprocessing import MinMaxScaler
baseball = MinMaxScaler().fit_transform(baseball)

# 2.Perform a cluster analysis on these players using k-means.
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)
kmeans.fit(baseball)
y_kmeans = kmeans.predict(baseball)

# 3.Look at a plot of WCSS versus number of clusters to help choose
# the optimal number of clusters. How many did you decide on? 

wcss = []
for k in range(1,16):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(baseball)
    wcss.append(kmeans.inertia_)

plt.plot(np.arange(1,16),wcss,marker="o")
plt.xlabel('Number of Clusters, K')
plt.ylabel('WCSS')
# looks like there is 3 groups with this elbow plot

# 4.Describe the properties that seem common to each cluster.
plt.scatter(baseball[:,0], baseball[:,1], c=y_kmeans, cmap='viridis',alpha=.5)
plt.scatter(baseball[:,1], baseball[:,2], c=y_kmeans, cmap='viridis',alpha=.5)
plt.scatter(baseball[:,2], baseball[:,3], c=y_kmeans, cmap='viridis',alpha=.5)
# It seems to be divided among rookie, average, and pro players.


# 5.Make a plot of the first two principal components colored by
#   predicted cluster label. Does it look like there is good separation between
#   the clusters?
plt.scatter(baseball[:,0], baseball[:,1], c=y_kmeans, cmap='viridis',alpha=.5)
# The separation is not very clear between clusters, but i'm only looking at
# two of the factors. it may be clearer between the factors. Either way, I don't
# think it could get much better.

### Part 2 ###

#files = (['Health-Tweets/bbchealth.txt',
                    #'Health-Tweets/cbchealth.txt',
                    #'Health-Tweets/cnnhealth.txt',
                    #'Health-Tweets/everydayhealth.txt',
                    #'Health-Tweets/foxnewshealth.txt',
                    #'Health-Tweets/gdnhealthcare.txt',
                    #'Health-Tweets/goodhealth.txt',
                    #'Health-Tweets/KaiserHealthNews.txt',
                    #'Health-Tweets/latimeshealth.txt',
                    #'Health-Tweets/msnhealthnews.txt',
                    #'Health-Tweets/NBChealth.txt',
                    #'Health-Tweets/nprhealth.txt', 
                    #'Health-Tweets/nytimeshealth.txt',
                    #'Health-Tweets/reuters_health.txt',
                    #'Health-Tweets/usnewshealth.txt',
                    #'Health-Tweets/wsjhealth.txt'])
#tweets = pd.DataFrame()

#for item in files:
    #info = pd.read_table(item, header = None, encoding = 'unicode_escape')
    #tweets = pd.concat([tweets, info], ignore_index = True)

file1 = pd.read_table('Health-Tweets/bbchealth.txt', header = None,
                        encoding = 'unicode_escape')
file1['label'] = 'bbchealth'

file2 = pd.read_table('Health-Tweets/wsjhealth.txt', header = None,
                        encoding = 'unicode_escape')
file2['label'] = 'wsjhealth'

tweets = pd.concat([file1, file2], ignore_index = True)

# 6. Clean the tweet text as you think most appropriate. 
#Combined Clearn
tweets['tweet_id'] = tweets[0].apply(lambda x: re.split('\|', str(x))[0])
tweets['date'] = tweets[0].apply(lambda x: re.split('\|', str(x))[1])
tweets['content'] = tweets[0].apply(lambda x: re.split('\|', str(x))[2])

tweets['date'] = tweets['date'].apply(lambda x: datetime.datetime.strptime(x,
                                                '%a %b %d %H:%M:%S +0000 %Y'))
tweets = tweets.drop([0], axis = 1)

tweets = tweets[['tweet_id', 'date', 'content', 'label']]

#file1
file1['tweet_id'] = file1[0].apply(lambda x: re.split('\|', str(x))[0])
file1['date'] = file1[0].apply(lambda x: re.split('\|', str(x))[1])
file1['content'] = file1[0].apply(lambda x: re.split('\|', str(x))[2])

file1['date'] = file1['date'].apply(lambda x: datetime.datetime.strptime(x,
                                                '%a %b %d %H:%M:%S +0000 %Y'))
file1 = file1.drop([0], axis = 1)

file1 = file1[['tweet_id', 'date', 'content', 'label']]


#file2
file2['tweet_id'] = file2[0].apply(lambda x: re.split('\|', str(x))[0])
file2['date'] = file2[0].apply(lambda x: re.split('\|', str(x))[1])
file2['content'] = file2[0].apply(lambda x: re.split('\|', str(x))[2])

file2['date'] = file2['date'].apply(lambda x: datetime.datetime.strptime(x,
                                                '%a %b %d %H:%M:%S +0000 %Y'))
file2 = file2.drop([0], axis = 1)

file2 = file2[['tweet_id', 'date', 'content', 'label']]



from nltk import wordpunct_tokenize, word_tokenize, sent_tokenize
from nltk.corpus import stopwords

sw = stopwords.words('english')

from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wn = WordNetLemmatizer()

def process_text(x):
    x = x.lower()
    tokens = wordpunct_tokenize(x)
    tokens = [tok for tok in tokens if tok.isalnum()]
    tokens = [tok for tok in tokens if tok not in sw]
    tokens = [wn.lemmatize(tok) for tok in tokens]
    return(tokens)

contents = tweets['content'].apply(process_text)
contents = contents.tolist()
contents = [item for sublist in contents for item in sublist]


# 7. What are the 10 most common words (that aren't stopwords) from the tweets?
from collections import Counter

c = Counter(contents)
wc = pd.DataFrame(c.items(), columns=['word','count'])
wc.sort_values(by='count',ascending=False).head(10)

# 8. What are the 10 most common words (that aren't stopwords) by news agency
#    (for just the two that you chose)?

#file1 (BBC Health)
contents_f1 = file1['content'].apply(process_text)
contents_f1 = contents_f1.tolist()
contents_f1 = [item for sublist in contents_f1 for item in sublist]

c = Counter(contents_f1)
wc = pd.DataFrame(c.items(), columns=['word','count'])
wc.sort_values(by='count',ascending=False).head(10)

#file2 (WSJ Health)
contents_f2 = file2['content'].apply(process_text)
contents_f2 = contents_f2.tolist()
contents_f2 = [item for sublist in contents_f2 for item in sublist]

c = Counter(contents_f2)
wc = pd.DataFrame(c.items(), columns=['word','count'])
wc.sort_values(by='count',ascending=False).head(10)

# 9. Perform a kmeans clustering on the text of the tweets (from both news
#    agencies).
from sklearn.feature_extraction.text import TfidfVectorizer

text = tweets['content'].apply(process_text)
text = text.apply(lambda x: " ".join(x))
tf = TfidfVectorizer()
X = tf.fit_transform(text)

Xdf = pd.DataFrame(X.toarray(), columns = tf.get_feature_names())

# takes a long time to run
wcss_tweets = []
for k in range(1,15):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    wcss_tweets.append(kmeans.inertia_)

plt.plot(np.arange(1,15),wcss_tweets,marker="o")
plt.xlabel('Number of Clusters, K')
plt.ylabel('WCSS')

from sklearn.cluster import MiniBatchKMeans
k = 6
kmeans = MiniBatchKMeans(k)
kmeans.fit(X)
labs = kmeans.predict(X)

# 10. How many clusters do you feel best summarizes the data?

# I felt like 6 clusters was the ideal number.

# 11. Interpret the clusters that you found by looking at the top words from each
#     cluster. 
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

terms = tf.get_feature_names()
for i in range(k):
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind], end='')
    print()


# 12. Build a model that uses the tweet text to classify which news agency
#     the tweet came from. How well is your model able to classify?
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
recall_score, roc_auc_score, roc_curve, accuracy_score)

train, test = train_test_split(tweets, test_size=.4, stratify=tweets.label, random_state=713)

tfidf = TfidfVectorizer(min_df = 50, stop_words = sw)
tfidf.fit(train['content'])
X_train = tfidf.transform(train['content'])
X_test = tfidf.transform(test['content'])

y_train = (train['label'] == 'wsjhealth').astype(int)
y_test = (test['label'] == 'wsjhealth').astype(int)

from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
nb.fit(X_train, y_train)
yhat = nb.predict(X_test)
y_prob = nb.predict_proba(X_test)[:,1]

print(roc_auc_score(y_test, y_prob))
print(accuracy_score(y_test, yhat))
print(f1_score(y_test, yhat))


