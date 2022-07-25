from ast import Str
from typing import final
from bs4 import BeautifulSoup
import re
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer,CountVectorizer
import sklearn.feature_extraction
from dateparser.search import search_dates

# def html2text(html):
#     with open(html, 'r') as f: 
#         html_string = f.read()
#         soup = BeautifulSoup(html_string,'html.parser')
#         ret = soup.get_text()
#         ret = re.sub(r"\s", " ", ret, flags = re.MULTILINE)
#         ret = re.sub("<br>|<br />|</p>|</div>|</h\d>", "\n", ret, flags = re.IGNORECASE)
#         ret = re.sub('<.*?>', ' ', ret, flags=re.DOTALL)
#         ret = re.sub(r"  +", " ", ret)
#     return ret


html = './templates/inbox-faith-email.html'
#html = './templates/inbox-rebecca-email.html'


def html2text(html):
    with open(html, 'r') as f: 
        html_string = f.read()
    soup = BeautifulSoup(html_string,'html.parser')

    for script in soup(["script", "style"]):
        script.extract()

    text = soup.body.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("."))
    text = " ".join(chunk for chunk in chunks if chunk)
    #meeting_details = search_dates(text)
    return text
#print(text)

text = html2text(html)

print(text)
meeting_details = search_dates(text)

print(meeting_details)

start_time = meeting_details[2][1].time().strftime('%H:%M')
end_time = meeting_details[3][1].time().strftime('%H:%M') 
date_faith = meeting_details[1][1].date().day
date = meeting_details[1][1].date().month 
print(start_time)
print(end_time)
print(date_faith)
print(date)

# email = '''
# Faith Tan <faith_tan@sutd.edu.sg> Thu 6/18/2020 4:49 PM To: You: A Dear Crystal, On behalf of the DAI Student Board, we would like to invite you to our first ever DAI bonding session. There will be nice music, amazing wine and splendid food accompanied with awesome human beings. We hope to see you there and spend an amaing afternoon together. Date: Friday, 29 July 2022Time: 2.00 pm to 6.00 pmVenue:           DAI Studio 7 What are you waiting for? Please RSVP via telegram by 23 July 2022. Thank you!Best regards, Faith Tan
# '''



from functools import reduce
from itertools import chain
from termcolor import colored

# HIGHLIGHT TERMS 

# def highlight_words(html):
    
#     text = html2text(html)
#     meeting_details = search_dates(text)
#     highlighted_terms = []
#     highlighted_terms.append(meeting_details[1][0])
#     highlighted_terms.append(meeting_details[2][0])
#     highlighted_terms.append(meeting_details[3][0])
#     highlight = print(reduce(lambda t, x: t.replace(*x), chain([text], ((t, colored(t,'white','on_blue')) for t in highlighted_terms))))
#     return highlight

# highlight_words('./templates/inbox-faith-email.html')

# para = [d for d in search_dates(text)]
# print(para)

# for i in meeting_details:
#     print(i[0][0])


# print(meeting_details)

# print(meeting_details[1][1].date().month)
# print(meeting_details[1][1].date().day)
# print(f"{meeting_details[1][1].date().day}/{meeting_details[1][1].date().month}")

# print(meeting_details[2][1].time().strftime('%H:%M'))
# print(meeting_details[3][1].time().strftime('%H:%M'))

# print(f"{meeting_details[2][1].time().hour}:{meeting_details[2][1].time().minute}")
# print(f"{meeting_details[3][1].time().hour}:{meeting_details[3][1].time().minute}")



# from datetime import datetime
# start_time = meeting_details[2][0]
# in_time = datetime.strptime(start_time, "%I %M ")
# out_time = datetime.strftime(in_time, "%H:%M")

# print(out_time)

# print(meeting_details[2][0])
# print(meeting_details[3][0])

# start_time = f"{meeting_details[2][1].time().strftime('%H:%M')}"
# end_time = f"{meeting_details[3][1].time().strftime('%H:%M')}"
# date = f"{meeting_details[1][1].date().day}/{meeting_details[1][1].date().month}/2022"


