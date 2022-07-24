# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import pickle
import requests
import json

# Importing the dataset
df = pd.read_csv('input621new.csv')
#X = df.iloc[:, :-1].values
vect = TfidfVectorizer(ngram_range=(1,3),
                     min_df = 20,
                     max_df = 0.5,
                     token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*[a-z]+\\w*\\b')
X = vect.fit_transform(df['Body'].values.astype('U'))

# Clustering the dataset
from sklearn.cluster import KMeans
n_clusters = 4
clf = KMeans(n_clusters=n_clusters, max_iter=300, init='k-means++', n_init=1)
labels = clf.fit_predict(X)

# top keywords in the emails
def top_tfidf_feats(row, features, top_n=20000):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats, columns=['features', 'score'])
    return df
def top_feats_in_doc(X, features, row_id, top_n=25):
    row = np.squeeze(X[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)

features = vect.get_feature_names()
top_feats_in_doc(X, features, 1, 10)

# top terms out of all the emails
def top_mean_feats(X, features, grp_ids=None, min_tfidf=0.1, top_n=20000):
    if grp_ids:
        D = X[grp_ids].toarray()
    else:
        D = X.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)
top_mean_feats(X, features, top_n=10)

# extracts the top terms per cluster
features = vect.get_feature_names()
def top_feats_per_cluster(X, y, features, min_tfidf=0.1, top_n=10):
    dfs = []
    print(y)
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label) 
        feats_df = top_mean_feats(X, features, ids,    min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs
top_feats_per_cluster(X, labels, features, min_tfidf=0.1, top_n=10)

#Use this to print the top terms per cluster with matplotlib.
def plot_tfidf_classfeats_h(dfs):
    fig = plt.figure(figsize=(25, 9), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(1, len(dfs), i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Tf-Idf Score", labelpad=16, fontsize=14)
        ax.set_title("cluster = " + str(df.label), fontsize=16)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.score, align='center', color='#7530FF')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        yticks = ax.set_yticklabels(df.features)
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.show()


plot_tfidf_classfeats_h(top_feats_per_cluster(X, labels, features, 0.1, 25))


df['Body'] = df['Body'].astype(str)
df['Body'] = df['Body'].apply(lambda x: x.lower())
punctuations = '\.\!\?'
df = (df.drop('Body',axis=1).merge(df.Body.str.extractall(f'(?P<Body>[^{punctuations}]+[{punctuations}])\s?').reset_index('match'),left_index=True, right_index=True, how='left'))
df['Body'] = df['Body'].str.replace("[^\w\s<>]", "")
df = df.replace(r'[^0-9a-zA-Z ]', '', regex=True).replace("'", '')
#print(type(df['Body']))
#df.applymap(type)
df['Body'] = df['Body'].astype(str)
df['Body'] = df['Body'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()])) # change similar terms to the same
# stop = stopwords.words("english") #remove useless words
df['Body'] = df['Body'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

df['Classifications'] = pd.Series(labels, index=df.index)
X = df['Body']
y = pd.to_numeric(df['Classifications'])

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Fitting Vectorizer to the Training set
vect = TfidfVectorizer(ngram_range=(1,3),
                     min_df = 5,
                     max_df = 0.4,
                     token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*[a-z]+\\w*\\b')
vect.fit(X_train)
X_train_dtm = vect.transform(X_train)
X_test_dtm = vect.transform(X_test)

# Changing from sparse to dense matrix
X_train_dense = pd.DataFrame(X_train_dtm.toarray(), columns = vect.get_feature_names_out())
X_test_dense = pd.DataFrame(X_test_dtm.toarray(), columns = vect.get_feature_names_out())
pca = PCA(n_components = 129)
X_train_pca = pca.fit_transform(X_train_dense)
X_test_pca = pca.transform(X_test_dense)

# the random forest algorithm randomly selects observations and features 
# to build several decision trees and then averages the results
# from sklearn.ensemble import RandomForestClassifier
# rfc = RandomForestClassifier(n_estimators=200, criterion="entropy", n_jobs=-1, random_state=0)
# rfc.fit(X_train_pca, y_train)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10,5),max_iter=500)
mlp.fit(X_train_pca,y_train)
y_pred_class = mlp.predict(X_test_pca)
#print(y_pred_class)


# a = [0,1,2,3]
# mapping = {0:'Meeting', 1:'Normal', 2:'Normal', 3:'Normal'}
# #a = [mapping[i] for i in a]
# #df['Classifications'] = df['Classifications'].map(a)
# df['Classifications'] = df['Classifications'].replace(mapping)
# df
# Predicting the Test set results
predictions = mlp.predict(X_test_pca)
#predictions = predictions.astype(int)

# res = [predictions[index].replace(1, 'Meeting') for index in range(len(predictions))]

# print(res)

# for i, n in enumerate(predictions):
#     print(n)
#     if n == 1:
#         predictions[i] = "Meeting"
#     else:
#         predictions[i] = "Normal"
# print(predictions)

# for index in predictions:
#     print(type(index))
#     if index == 3:
#         index == "Meeting"
#     else:
#         index == "Normal"
# print(predictions)

# for index in range(len(predictions)):
#     print(type(index))
#     if predictions[index] == 3:
#         predictions[index] == "Meeting"
#     else:
#         predictions[index] == "Normal"

# print(predictions)



# Saving model to disk
pickle.dump(mlp, open('model.pkl','wb'))
# Loading model to compare the results
#model = pickle.load(open('model.pkl','rb'))
#print(model.predict(X_test_pca))