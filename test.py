import requests
import time

url = 'http://localhost:5000/compare'
file_path = 'src/db/test_faces/3.jpg'
cookies = {} # needed for codespace
with open(file_path, 'rb') as f:
    files = {'photo': f}
    start_time = time.time()
    response = requests.post(url, files=files, cookies=cookies)
    end_time = time.time()

print(response)
print(response.json())
print("Response time: {} seconds".format(end_time - start_time))