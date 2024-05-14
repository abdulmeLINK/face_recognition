from flask import Flask, request, jsonify
from facenet.model import FaceNetModel
import numpy as np
from utils.gpu_utils import check_gpu_availability, configure_gpu
import os
import cv2

app = Flask(__name__)

if check_gpu_availability():
    configure_gpu()

# Load the FaceNet model
model = FaceNetModel()

@app.route('/compare', methods=['POST'])
def compare_faces():
    # Get the photo to compare from the request
    photo = request.files['photo']

    # Preprocess the photo and compute its embedding
    embedding = model.compute_embedding(photo)

    # Compare the embedding to the faces in the database
    match = compare_to_database(embedding)

    # Return the result as JSON
    return jsonify({'match': match})



def load_embeddings_from_database():
    # Define the path to the database
    database_path = os.path.join(os.path.dirname(__file__), '../db/faces')

    # Initialize an empty list to store the embeddings
    database_embeddings = []

    # Iterate over the images in the database
    for filename in os.listdir(database_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Load the image
            image_path = os.path.join(database_path, filename)
            image = cv2.imread(image_path)

            # Preprocess the image and compute its embedding
            embedding = model.compute_embedding(image)

            # Add the embedding to the list
            database_embeddings.append(embedding)

    return database_embeddings

def calculate_euclidean_distance(embedding1, embedding2):
    return np.sqrt(np.sum((embedding1 - embedding2) ** 2))

def compare_to_database(embedding):
    database_embeddings = load_embeddings_from_database()

    for db_embedding in database_embeddings:
        distance = calculate_euclidean_distance(embedding, db_embedding)
        if distance < 0.5:  # This threshold may need to be adjusted based on your specific use case
            return True

    return False

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)