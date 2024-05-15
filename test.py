import requests

url = 'http://localhost:5000/compare'
file_path = 'src/db/test_faces/3.jpg'

with open(file_path, 'rb') as f:
    files = {'photo': f}
    response = requests.post(url, files=files)

print(response.json())