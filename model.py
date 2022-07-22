# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import pickle
import requests
import json
# Importing the dataset
df = pd.read_csv('input500.csv')
X = df.iloc[:, :-1].values
vect = TfidfVectorizer(ngram_range=(1,3),
                     min_df = 10,
                     max_df = 0.4,
                     token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*[a-z]+\\w*\\b')
X = vect.fit_transform(df['Body'].values.astype('U'))

# Clustering the dataset
from sklearn.cluster import KMeans
n_clusters = 6
clf = KMeans(n_clusters=n_clusters, max_iter=300, init='k-means++', n_init=1)
labels = clf.fit_predict(X)
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
pca = PCA(n_components = 128)
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
print(y_pred_class)


# Predicting the Test set results
predictions = mlp.predict(X_test_pca)
print(predictions)

# Saving model to disk
pickle.dump(mlp, open('model.pkl','wb'))
# Loading model to compare the results
#model = pickle.load(open('model.pkl','rb'))
#print(model.predict(X_test_pca))