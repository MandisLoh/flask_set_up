from flask import Flask
from flask import render_template
from datetime import datetime
import re

app = Flask(__name__)

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

# The decorator used for the new URL route, /hello/<name>, defines an endpoint /hello/ that can accept any additional value. 
# The identifier inside < and > in the route defines a variable that is passed to the function and can be used in your code.

@app.route("/hello/")
@app.route("/hello/<name>")
def hello_there(name = None):
    return render_template(
        "hello_there.html",
        name=name,
        date=datetime.now()
    )

@app.route("/api/data")
def get_data():
    return app.send_static_file("data.json")



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