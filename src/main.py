import os
import cv2
from facenet.model import FaceNetModel
from service.face_comparator import load_embeddings_from_database, calculate_euclidean_distance, compare_to_database

# Load the FaceNet model
model = FaceNetModel()
model.load_model()

# Load the embeddings from the database
database_embeddings, filenames, database_tree = load_embeddings_from_database(model)

# Process the input photo and compare it to the embeddings in the database
def compare_faces(input_photo):
    input_face = cv2.imread(input_photo)
    input_embedding = model.compute_embedding(input_face)
    result = compare_to_database(input_embedding, database_embeddings, filenames, database_tree)
    return result

# Example usage
input_photo = 'db/test_faces/1.jpg'
result = compare_faces(input_photo)
print(result)