import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint

import nltk
from nltk import wordpunct_tokenize, word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wn = WordNetLemmatizer()
sw = stopwords.words('english')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                recall_score, confusion_matrix, roc_auc_score,
                                roc_curve)

def text_token(x):
    x = x.lower()
    tokens = wordpunct_tokenize(x)
    tokens = [tok for tok in tokens if tok.isalnum()]
    tokens = [tok for tok in tokens if tok not in sw]
    tokens = [wn.lemmatize(tok) for tok in tokens]
    return(tokens)

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

# 1. Import the data. Note that the dataset is tab delimited.
ratings = pd.read_csv('ratings.tsv', delimiter = '\t')
ratings.head()

# 2. Check for and remove missing values and blank strings
ratings['review'].replace('', np.nan, inplace=True)
ratings.dropna(subset=['review'], inplace=True)

# 3. Split the data into a training set and a test set. Use test size=0.33,
# stratify=y, and random state=801 (where y is the label positive or negative)
train, test = train_test_split(ratings, test_size=.33, stratify=ratings.label,
                                random_state=801)

y_train = (train['label']=='pos').astype(int)
y_test = (test['label']=='pos').astype(int)

# 4. Vectorize the data using TD-IDF. Be sure that all model development is with
# the training data (fit the TD-IDF transformer on the training data, then
#         transform to both training and test data).

tfidf = TfidfVectorizer(stop_words = sw)
tfidf.fit(train['review'])
X_train = tfidf.transform(train['review'])
X_test = tfidf.transform(test['review'])

# 5. Build a machine learning classifier. Try out various models including:

# • Support Vector Classifier
from sklearn.svm import SVC

svc = SVC(probability=True, kernel='linear')
svc.fit(X_train, y_train)

yhat_svc = svc.predict(X_test)

print("Accuracy Score")
print(accuracy_score(y_test, yhat_svc))
print("\n F1 Score")
print(f1_score(y_test, yhat_svc))
print("\n Precision Score")
print(precision_score(y_test, yhat_svc))
print("\n Recall Score")
print(recall_score(y_test, yhat_svc))

y_prob = svc.predict_proba(X_test)[:,1]
fpr, tpr, thresh = roc_curve(y_test, y_prob)
plot_roc_curve(fpr, tpr)

print(roc_auc_score(y_test, y_prob))

# • Multilayer Perceptron
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2),
                    random_state=1)

mlp.fit(X_train, y_train)

yhat_mlp = mlp.predict(X_test)

print("Accuracy Score")
print(accuracy_score(y_test, yhat_mlp))
print("\n F1 Score")
print(f1_score(y_test, yhat_mlp))
print("\n Precision Score")
print(precision_score(y_test, yhat_mlp))
print("\n Recall Score")
print(recall_score(y_test, yhat_mlp))

y_prob = mlp.predict_proba(X_test)[:,1]
fpr, tpr, thresh = roc_curve(y_test, y_prob)
plot_roc_curve(fpr, tpr)

print(roc_auc_score(y_test, y_prob))

# • Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()

nb.fit(X_train, y_train)

yhat = nb.predict(X_test)

print("Accuracy Score")
print(accuracy_score(y_test, yhat))
print("\n F1 Score")
print(f1_score(y_test, yhat))
print("\n Precision Score")
print(precision_score(y_test, yhat))
print("\n Recall Score")
print(recall_score(y_test, yhat))

y_prob = nb.predict_proba(X_test)[:,1]
fpr, tpr, thresh = roc_curve(y_test, y_prob)
plot_roc_curve(fpr, tpr)

print(roc_auc_score(y_test, y_prob))

# • (You are welcome to try other models if you want)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_depth = 100, n_estimators = 500, n_jobs = -1)
rf.fit(X_train, y_train)
yhat_rf = (rf.predict(X_test))

print("Accuracy Score")
print(accuracy_score(y_test, yhat_rf))
print("\n F1 Score")
print(f1_score(y_test, yhat_rf))
print("\n Precision Score")
print(precision_score(y_test, yhat_rf))
print("\n Recall Score")
print(recall_score(y_test, yhat_rf))

y_prob = rf.predict_proba(X_test)[:,1]
fpr, tpr, thresh = roc_curve(y_test, y_prob)
plot_roc_curve(fpr, tpr)

print(roc_auc_score(y_test, y_prob))

# 6. What model performs the best? Experiment with changing the hyper-parameters,
# changing the vectorizer, adding bi-grams and/or using a voting classifier to
# increase model accuracy.

# So based on the various metrics it looks like Naive Bayes was the winner.
# It clearly had the best scores, though Random Forest was close as well.

# 7. If possible, report what words are most important for distinguishing between
# positive and negative reviews?

feats = pd.DataFrame()
feats['words'] = tfidf.get_feature_names()
# Convert log probabilities to probabilities. 
feats['neg'] = np.e**(nb.feature_log_prob_[0, :])
feats['pos'] = np.e**(nb.feature_log_prob_[1, :])
feats.set_index('words', inplace=True)

feats.sort_values(by='neg',ascending=False).head(15)

feats.sort_values(by='pos',ascending=False).head(15)

# 8. Using Vader sentiment analysis, predict whether or not the movie review is
# positive or negative. (Use a positive compound score for “positive” and a
#         negative compound score for “negative”).
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

ratings['scores'] = ratings['review'].apply(lambda x: sid.polarity_scores(x))
ratings['compound'] = ratings['scores'].apply(lambda x: x['compound'])

ratings.head(50)

#it seems to have pretty well, getting a majority of them correct.

# 9. How does the accuracy of the sentiment analysis compare with that of the
# predictive model?
sns.boxplot(x='label', y='compound',data=ratings)

# 10. Try doing sentiment analysis with the TextBlob library. How does the
# accuracy of TextBlob sentiments compare with Vader and the predictive model?
from textblob import TextBlob

def sentiment_calc(text):
    try:
        return TextBlob(text).sentiment
    except:
        return None
    
ratings['sentiment'] = ratings['label'].apply(sentiment_calc)

print(sentiment_calc(ratings['review'][1]))
print(sentiment_calc(ratings['review'][2]))
print(sentiment_calc(ratings['review'][3]))
print(sentiment_calc(ratings['review'][4]))
print(sentiment_calc(ratings['review'][5]))
print(sentiment_calc(ratings['review'][6]))
print(sentiment_calc(ratings['review'][7]))
print(sentiment_calc(ratings['review'][8]))
print(sentiment_calc(ratings['review'][9]))
print(sentiment_calc(ratings['review'][10]))

#Textblob does much worse than vader.

# 11. Run LDA topic modeling using gensim on the movie reviews. How many topics
# are there? What are the most common words in each topic?
import gensim, spacy, logging, warnings
import gensim.corpora as corpora
from gensim.utils import lemmatize, simple_preprocess
from gensim.models import CoherenceModel

reviews = ratings['review'].apply(lambda x:
        gensim.utils.simple_preprocess(str(x), deacc = True))

# Build the bigram and trigram models
bigram = gensim.models.Phrases(reviews, min_count=5, threshold=10) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[reviews], threshold=10)  
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

def process_words(texts, stop_words=sw, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """Remove Stopwords, Form Bigrams, Trigrams and perform Lemmatization"""
    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    
    texts_out = []
    nlp = spacy.load('en', disable=['parser', 'ner'])    # Load spacy, but we don't need the parser or NER (named entity extraction) modules
    
    for sent in texts:
        doc = nlp(" ".join(sent)) # perform tokenization, part of speech tagging, etc. on the concatenated words
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        
    # remove stopwords once more after lemmatization
    texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]    
    return texts_out

reviews = process_words(reviews)

doc_lens = [len(d) for d in reviews]

# Plot
plt.figure(figsize=(16,7), dpi=160)
plt.hist(doc_lens, bins = 1000, color='navy')
plt.text(750, 100, "Mean   : " + str(round(np.mean(doc_lens))))
plt.text(750,  90, "Median : " + str(round(np.median(doc_lens))))
plt.text(750,  80, "Stdev   : " + str(round(np.std(doc_lens))))
plt.text(750,  70, "1%ile    : " + str(round(np.quantile(doc_lens, q=0.01))))
plt.text(750,  60, "99%ile  : " + str(round(np.quantile(doc_lens, q=0.99))))

plt.gca().set(xlim=(0, 1000), ylabel='Number of Documents', xlabel='Document Word Count')
plt.tick_params(size=16)
plt.xticks(np.linspace(0,1000,9))
plt.title('Distribution of Document Word Counts', fontdict=dict(size=22))
plt.show()

ntopics = 12
# There were 12 topics

# Create Dictionary
id2word = corpora.Dictionary(reviews)

# Create Corpus: Term Document Frequency
corpus = [id2word.doc2bow(text) for text in reviews]

# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=ntopics, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=10,
                                           passes=10,
                                           alpha='symmetric',
                                           iterations=100,
                                           per_word_topics=True)

ldatopics = lda_model.show_topics(formatted=False)
pprint(lda_model.print_topics())

# Visualize the topic model
import pyLDAvis.gensim

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)
#vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word, mds='mmds')
vis

def format_topics_sentences(ldamodel=None, corpus=corpus, texts=reviews):
  sent_topics_df = pd.DataFrame()

  for i, doc in enumerate(corpus):
    topics = lda_model.get_document_topics(doc)
    topics = sorted(topics, key=lambda x: (x[1]), reverse=True)
    dominant_topic = topics[0][0]
    dominant_topic_prob = round(topics[0][1], 4)
    dominant_topic_keywords_probs = lda_model.show_topic(topics[0][0])
    dominant_topic_keywords = ", ".join([word for word, prob in dominant_topic_keywords_probs])
  
    sent_topics_df = sent_topics_df.append(pd.Series([ int(i), int(dominant_topic), dominant_topic_prob, dominant_topic_keywords, reviews[i] ]), ignore_index=True)
  
  sent_topics_df.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Topic_Keywords', 'Text']
  sent_topics_df = sent_topics_df.astype({'Document_No': int, 'Dominant_Topic': int})
  return sent_topics_df

df_dominant_topic = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=reviews)

# Display setting to how more characters in column
pd.options.display.max_colwidth = 100
df_dominant_topic.head(10)

data = ratings['review'].tolist()

print("Document #:", df_dominant_topic.loc[0]['Document_No'])
print("Dominant Topic:", df_dominant_topic.loc[0]['Dominant_Topic'])
print("Topic Percent Contribution:", df_dominant_topic.loc[0]['Topic_Perc_Contrib'])
print("Topic Keywords:", df_dominant_topic.loc[0]['Topic_Keywords'])
print("Text:", df_dominant_topic.loc[0]['Text'])
print("\n_________Data __________\n", data[0])

df_dominant_topic_grpd = df_dominant_topic.groupby('Dominant_Topic')

df_group_rep = pd.DataFrame()

for i, grp in df_dominant_topic_grpd:
    df_group_rep = pd.concat([df_group_rep, grp.sort_values(['Topic_Perc_Contrib'], ascending=False).head(1)], axis=0)

# Reset Index    
df_group_rep.reset_index(drop=True, inplace=True)

# Format
df_group_rep.columns = ['Document_Num', 'Topic_Num', "Topic_Perc_Contrib", "Topic_Keywords", "Representative Text"]

# Show
df_group_rep.head(10)

doc = 8
print("Document #:", df_dominant_topic.loc[doc]['Document_No'])
print("Dominant Topic:", df_dominant_topic.loc[doc]['Dominant_Topic'])
print("Topic Percent Contribution:", df_dominant_topic.loc[doc]['Topic_Perc_Contrib'])
print("Topic Keywords:", df_dominant_topic.loc[doc]['Topic_Keywords'])
print("Text:", df_dominant_topic.loc[doc]['Text'])
print("\n_________Data __________\n", data[doc])

#Distribution of words for each topic
import matplotlib.colors as mcolors
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

fig, axes = plt.subplots(3,3,figsize=(16,14), dpi=160, sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):    
    df_dominant_topic_sub = df_dominant_topic.loc[df_dominant_topic.Dominant_Topic == i, :]
    doc_lens = [len(d) for d in df_dominant_topic_sub.Text]
    ax.hist(doc_lens, bins = 1000, color=cols[i])
    ax.tick_params(axis='y', labelcolor=cols[i], color=cols[i])
    sns.kdeplot(doc_lens, color="black", shade=False, ax=ax.twinx())
    ax.set(xlim=(0, 1000), xlabel='Document Word Count')
    ax.set_ylabel('Number of Documents', color=cols[i])
    ax.set_title('Topic: '+str(i), fontdict=dict(size=16, color=cols[i]))

fig.tight_layout()
fig.subplots_adjust(top=0.90)
plt.xticks(np.linspace(0,1000,9))
fig.suptitle('Distribution of Document Word Counts by Dominant Topic', fontsize=22)
plt.show()

# 1. Wordcloud of Top N words in each topic
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(stopwords=sw,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = lda_model.show_topics(formatted=False)

fig, axes = plt.subplots(3, 3, figsize=(10,10), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')


plt.subplots_adjust(wspce=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()

# Get topic weights and dominant topics ------------
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook

# Get topic weights
topic_weights = []
for i, row_list in enumerate(lda_model[corpus]):
    topic_weights.append([w for i, w in row_list[0]])

# Array of topic weights    
arr = pd.DataFrame(topic_weights).fillna(0).values

# Keep the well separated points (optional)
#arr = arr[np.amax(arr, axis=1) > 0.35]

# Dominant topic number in each doc
topic_num = np.argmax(arr, axis=1)

# tSNE Dimension Reduction
tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
tsne_lda = tsne_model.fit_transform(arr)

# Plot the Topic Clusters using Bokeh
output_notebook()
n_topics = 4
mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
plot = figure(title="t-SNE 2D Plot of {} LDA Topics".format(n_topics), 
              plot_width=900, plot_height=700)
plot.scatte(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_num])
show(plot)

