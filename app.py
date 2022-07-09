from flask import Flask
from flask import render_template
from datetime import datetime
import re

app = Flask(__name__)

# @app.route("/")
# def home():
#     return render_template("home.html")
# Replace the existing home function with the one below

@app.route("/")
def file():
    return render_template("file.html")

@app.route("/home/")
def home():
    return render_template("home.html")

@app.route("/sendreceive/")
def sendreceive():
    return render_template("sendreceive.html")



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