from flask import Flask, request, jsonify
from facenet.model import FaceNetModel
import numpy as np
from service.face_comparator import load_embeddings_from_database,compare_to_database, compare_to_database_cosine
from utils.gpu_utils import check_gpu_availability, configure_gpu   
import os
import cv2

app = Flask(__name__)

if check_gpu_availability():
    configure_gpu()

# Load the FaceNet model
model = FaceNetModel()
model.load_model()
database_embeddings, filenames, database_tree, cosine_distances = load_embeddings_from_database(model)

@app.route('/compare', methods=['POST'])
def compare_faces():
    # Get the photo to compare from the request
    photo = request.files['photo']

    # Preprocess the photo and compute its embedding
    embedding = model.compute_embedding(photo)

    # Compare the embedding to the faces in the database
    match = compare_to_database_cosine(embedding, database_embeddings, filenames)

    # Return the result as JSON
    return jsonify({'match': match})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)