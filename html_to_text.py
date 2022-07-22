from typing import final
from bs4 import BeautifulSoup
import re
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer,CountVectorizer
import sklearn.feature_extraction

def html2text(html):
    with open(html, 'r') as f: 
        html_string = f.read()
        soup = BeautifulSoup(html_string,'html.parser')
        ret = soup.get_text()
        ret = re.sub(r"\s", " ", ret, flags = re.MULTILINE)
        ret = re.sub("<br>|<br />|</p>|</div>|</h\d>", "\n", ret, flags = re.IGNORECASE)
        ret = re.sub('<.*?>', ' ', ret, flags=re.DOTALL)
        ret = re.sub(r"  +", " ", ret)
    return ret


# html = './templates/inbox-faith-email.html'

# model = pickle.load(open('model.pkl', 'rb'))
# count_vect = CountVectorizer()
# final_features = html2text(html)
# prediction = model.predict(count_vect.fit_transform([final_features]))

# print(prediction)
