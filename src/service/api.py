from flask import Flask, request, jsonify
from facenet.model import FaceNetModel
from werkzeug.datastructures import FileStorage
from PIL import Image
import numpy as np
import numpy as np
from service.face_comparator import load_embeddings_from_database,get_face_detector, compare_to_database_cosine
import os
import cv2

app = Flask(__name__)

# Load the FaceNet model
model = FaceNetModel()
model.load_model()
database_embeddings, filenames, database_tree, cosine_distances = load_embeddings_from_database(model)

@app.route('/compare', methods=['POST'])
def compare_faces():
    # Get the photo to compare from the request
    photo: FileStorage = request.files['photo']

    # Convert the photo to an image
    image = Image.open(photo.stream)
    image = np.array(image)

    # Detect faces in the image
    detector = get_face_detector()
    faces = detector(image, 1)

    matches = []
    for i, face in enumerate(faces):
        # Extract the region of interest from the image
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # Check that the coordinates are within the bounds of the image
        if x >= 0 and y >= 0 and x + w <= image.shape[1] and y + h <= image.shape[0]:
            roi = image[y:y+h, x:x+w]

            # Preprocess the face and compute its embedding
            embedding = model.compute_embedding(roi)

            # Compare the embedding to the faces in the database
            match = compare_to_database_cosine(embedding, database_embeddings, filenames)
            matches.append({'face': i, 'match': match})

    # Return the result as JSON
    return jsonify(matches)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)