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
from html_to_text import html2text
from dateparser.search import search_dates


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

# #To use the predict button in our web-app
@app.route('/predict',methods=['GET'])
def predict():
   html = './templates/inbox-faith-email.html'
   count_vect = CountVectorizer()
   final_features = html2text(html)
   prediction = model.predict(count_vect.fit_transform([final_features]))
   
   if (prediction==3):
      prediction = "Meeting"
   else:
      prediction = "Normal"
   return render_template('predict.html', prediction_text=f'Predicted category is {prediction}')

@app.route('/')
def inbox_main_unread():
   return render_template('inbox-main-unread.html')

@app.route('/inbox-faith-email-detected.html/inbox-faith-email.html/')
def inbox_faith_email():
   return render_template('inbox-faith-email.html')

@app.route('/inbox-faith-email-detected.html/inbox-faith-email.html/inbox-faith-email-w-pop-up.html')
def inbox_faith_email_w_pop_up():
   text = html2text('./templates/inbox-faith-email.html')
   meeting_details = search_dates(text)
   fh= open('./templates/calendar-main.html')
   content = fh.read()
   content = content.replace(re.findall('''<div class="caption-UdZOvo valign-text-middle opensans-bold-chicago-16px">Friday</div>
         <div class="caption-Q8xn1u valign-text-middle opensans-bold-chicago-16px">24</div>''', content)[0], 
         '''<div class="caption-UdZOvo valign-text-middle opensans-bold-chicago-16px">Friday</div>
         <div class="caption-Q8xn1u valign-text-middle opensans-bold-chicago-16px">24</div>
         <br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br />
         <div class="event-2-8ljLLo">
            <div class="rectangle-60-vu7vef border-1px-jordy-blue"></div>
               <div class="caption-vu7vef valign-text-middle opensans-bold-chicago-16px">DAI Bonding Session</div>
               <div class="caption-wx0sS7 valign-text-middle opensans-normal-chicago-12px">
            faith_tan@sutd.edu.sg<br />Studio 7
            </div> 
         </div>''' )
   fh.close()
   fh=open('./templates/calendar-main.html','w')
   fh.write(content)
   return render_template('inbox-faith-email-w-pop-up.html', start_time = f"{meeting_details[2][1].time().strftime('%H:%M')}", end_time = f"{meeting_details[3][1].time().strftime('%H:%M')}", date_faith = f"{meeting_details[1][1].date().day}/{meeting_details[1][1].date().month}/2022")

@app.route('/inbox-faith-email-detected.html')
def inbox_faith_email_detected():
   html = './templates/inbox-faith-email.html'
   count_vect = CountVectorizer()
   final_features = html2text(html)
   prediction = model.predict(count_vect.fit_transform([final_features]))
   
   if (prediction==0) or (prediction==3):
      prediction = "Meeting"
      
   else:
      prediction = "Normal"
   return render_template('inbox-faith-email-detected.html', email =prediction)

@app.route('/inbox-rebecca-email-detected.html/inbox-rebecca-email.html')
def inbox_rebecca_email():
   return render_template('inbox-rebecca-email.html')

@app.route('/inbox-rebecca-email-detected.html/inbox-rebecca-email.html/inbox-rebecca-email-w-pop-up.html')
def inbox_rebecca_email_w_pop_up():
   text = html2text('./templates/inbox-rebecca-email.html')
   meeting_details = search_dates(text)
   fh= open('./templates/calendar-main.html')
   content = fh.read()
   content = content.replace(re.findall('''<div class="caption-Cexj9x valign-text-middle opensans-bold-chicago-16px">Tuesday</div>
         <div class="caption-ebNaBO valign-text-middle opensans-bold-chicago-16px">21</div>''', content)[0],
         '''<div class="caption-Cexj9x valign-text-middle opensans-bold-chicago-16px">Tuesday</div>
         <div class="caption-ebNaBO valign-text-middle opensans-bold-chicago-16px">21</div>
         <br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br />
         <div class="event-2-s35V3b">
            <div class="rectangle-60-1KUmQd border-1px-jordy-blue">
               <div class="caption-1KUmQd valign-text-middle opensans-bold-chicago-16px">Annual General Meeting</div>
               <div class="caption-kYQchB valign-text-middle opensans-normal-chicago-12px">
            rebecca_tan@sutd.edu.sg<br />Think Thank 23 (2.413)
               </div> 
            </div>
         </div>''' )
   fh.close()
   fh=open('./templates/calendar-main.html','w')
   fh.write(content)
   return render_template('inbox-rebecca-email-w-pop-up.html', start_time = f"{meeting_details[2][1].time().strftime('%H:%M')}", end_time = f"{meeting_details[3][1].time().strftime('%H:%M')}", dates = f"{meeting_details[1][1].date().day}/{meeting_details[1][1].date().month}/2022")

@app.route('/inbox-rebecca-email-detected.html')
def inbox_rebecca_email_detected():
   html = './templates/inbox-rebecca-email.html'
   count_vect = CountVectorizer()
   final_features = html2text(html)
   prediction = model.predict(count_vect.fit_transform([final_features]))
   
   if (prediction==0) or (prediction==3):
      prediction = "Meeting"
      
   else:
      prediction = "Normal"
   return render_template('inbox-rebecca-email-detected.html', email =prediction)


@app.route('/inbox-faith-email-detected.html/inbox-faith-email.html/calendar-event-3.html')
def calendar_event_3():
   return render_template('calendar-event-3.html')

@app.route('/inbox-rebecca-email-detected.html/inbox-rebecca-email.html/calendar-event-2.html')
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

@app.route('/calendar-event-123.html')
def calendar_event_123():
   return render_template('calendar-event-123.html')

@app.route('/calendar-main.html')
def calendar_main():
   return render_template('calendar-main.html')

@app.route('/inbox-faith-email-detected.html/inbox-faith-email.html/calendar-event-123.html')
def calendar_event_123_1():
   return render_template('calendar-event-123.html')

@app.route('/inbox-rebecca-email-detected.html/inbox-rebecca-email.html/calendar-event-123.html')
def calendar_event_123_2():
   return render_template('calendar-event-123.html')

@app.route('/reset')
def resetting():
   fh= open('./templates/calendar-main1.html')
   content = fh.read()
   fh.close()
   fh=open('./templates/calendar-main.html','w')
   fh.write(content)
   return 'calendar reset'


if __name__ == "__main__":
   app.run(debug=True)
