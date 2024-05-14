import os
import cv2
import numpy as np
from facenet.model import FaceNetModel
from service.api import FaceComparisonService

# Load the FaceNet model
model = FaceNetModel("FaceNet512")

# Load the faces from the database
database_path = os.path.join(os.path.dirname(__file__), 'db/faces')
database_faces = []
for filename in os.listdir(database_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(database_path, filename)
        face = cv2.imread(image_path)
        database_faces.append(face)

# Initialize the face comparison service
service = FaceComparisonService(model, database_faces)  

# Process the input photo and compare it to the faces in the database
def compare_faces(input_photo):
    input_face = cv2.imread(input_photo)
    result = service.compare(input_face)
    return result

# Example usage
input_photo = 'db/test_faces/test.jpg'
result = compare_faces(input_photo)
print(result)