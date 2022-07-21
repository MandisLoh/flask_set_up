from flask import Flask
from flask import render_template
from datetime import datetime
import re
#import libraries
import numpy as np
import pandas as pd
from flask import Flask, render_template,request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
# @app.route("/")
# def home():
#     return render_template("home.html")
# Replace the existing home function with the one below

@app.route('/')
def inbox_main_unread():
   return render_template('inbox-main-unread.html')

@app.route('/inbox-faith-email.html/')
def inbox_faith_email():
   return render_template('inbox-faith-email.html')

@app.route('/inbox-faith-email.html/inbox-faith-email-w-pop-up.html')
def inbox_faith_email_w_pop_up():
   return render_template('inbox-faith-email-w-pop-up.html')

@app.route('/inbox-faith-email.html/calendar-event-2.html')
def calendar_event_2():
   return render_template('calendar-event-2.html')

@app.route('/inbox-new-email.html')
def inbox_new_email():
   return render_template('inbox-new-email.html')

@app.route('/new-e-mail-message-empty.html')
def new_e_mail_message_empty():
   return render_template('new-e-mail-message-empty.html')

@app.route('/new-e-mail-message.html')
def new_e_mail_message():
   return render_template('new-e-mail-message.html')

@app.route('/new-e-mail-message-w-pop-up.html')
def new_e_mail_message_w_pop_up():
   return render_template('new-e-mail-message-w-pop-up.html')

@app.route('/new-appointment.html')
def new_appointment():
   return render_template('new-appointment.html')

@app.route('/calendar-event-1.html')
def calendar_event_1():
   return render_template('calendar-event-1.html')

@app.route('/calendar-event-12.html')
def calendar_event_12():
   return render_template('calendar-event-12.html')

@app.route('/calendar-main.html')
def calendar_main():
   return render_template('calendar-main.html')

@app.route('/inbox-faith-email.html/calendar-event-12.html')
def calendar_event_12_1():
   return render_template('calendar-event-12.html')

# #To use the predict button in our web-app
@app.route('/predict.html',methods=['GET'])
def predict():
      df = pd.read_csv('testing100.csv')
      vect = TfidfVectorizer(ngram_range=(1,3),
               min_df = 10,
               max_df = 0.4,
               token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*[a-z]+\\w*\\b')
      X = vect.fit_transform(df['Body'].values.astype('U'))
      from sklearn.cluster import KMeans
      n_clusters = 6
      clf = KMeans(n_clusters=n_clusters, max_iter=300, init='k-means++', n_init=1)
      labels = clf.fit_predict(X)
      df['Classifications'] = pd.Series(labels, index=df.index)
      X = df['Body']
      y = pd.to_numeric(df['Classifications'])

      # Splitting the dataset into the Training set and Test set
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
      # Fitting Simple Linear Regression to the Training set
      vect = TfidfVectorizer(ngram_range=(1,3),
                  min_df = 5,
                  max_df = 0.4,
                  token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*[a-z]+\\w*\\b')
      vect.fit(X_train)
      X_train_dtm = vect.transform(X_train)
      X_test_dtm = vect.transform(X_test)

      X_train_dense = pd.DataFrame(X_train_dtm.toarray(), columns = vect.get_feature_names())
      X_test_dense = pd.DataFrame(X_test_dtm.toarray(), columns = vect.get_feature_names())
      pca = PCA(n_components = 20)
      X_train_pca = pca.fit_transform(X_train_dense)
      X_test_pca = pca.transform(X_test_dense)

      from sklearn.ensemble import RandomForestClassifier
      rfc = RandomForestClassifier(n_estimators=20, criterion="entropy", n_jobs=-1, random_state=0)
      rfc.fit(X_train_pca, y_train)
      predictions = rfc.predict(X_test_pca)

      #For rendering results on HTML GUI
      int_features = [float(x) for x in request.form.values()]
      final_features = X_test_pca
      prediction = model.predict(final_features)
      output = round(prediction[0]) 
   #return render_template('predict.html')
      return render_template('predict.html', prediction_text='Predicted output is {}'.format(output))




# The decorator used for the new URL route, /hello/<name>, defines an endpoint /hello/ that can accept any additional value. 
# The identifier inside < and > in the route defines a variable that is passed to the function and can be used in your code.

# @app.route("/hello/")
# @app.route("/hello/<name>")
# def hello_there(name = None):
#     return render_template(
#         "hello_there.html",
#         name=name,
#         date=datetime.now()
#     )

# @app.route("/api/data")
# def get_data():
#     return app.send_static_file("data.json")



# @app.route("/hello/<name>")
# def hello_there(name):
#     now = datetime.now()

#     # now.strftime("%a, %d %B, %Y at %X") -> 'Wed, 31 October, 2018 at 18:13:39'
#     # now.strftime("%a, %d %b, %Y at %X") -> 'Wed, 31 Oct, 2018 at 18:13:39'
#     # now.strftime("%a, %d %b, %y at %X") -> 'Wed, 31 Oct, 18 at 18:13:39'

#     formatted_now = now.strftime("%A, %d %B, %Y at %X")

#     # Filter the name argument to letters only using regular expressions. URL arguments
#     # can contain arbitrary text, so we restrict to safe characters only.
#     match_object = re.match("[a-zA-Z]+", name)

#     if match_object:
#         clean_name = match_object.group(0)
#     else:
#         clean_name = "Friend"

#     content = "Hello there, " + clean_name + "! It's " + formatted_now
#     return content