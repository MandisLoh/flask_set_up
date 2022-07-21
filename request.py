import requests
url = 'http://localhost:5000/api'
r = requests.post(url,json={"Body":"hello we meeting at 2pm",})
print(r.json())