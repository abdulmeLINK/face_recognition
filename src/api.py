from flask import Flask, request, jsonify
from facenet.model import FaceNetModel
from werkzeug.datastructures import FileStorage
from PIL import Image
import numpy as np
import numpy as np
from service.face_comparator import load_embeddings_from_database,get_face_detector, compare_to_database_cosine
import time


app = Flask(__name__)

# Load the FaceNet model
model = FaceNetModel()
model.load_model()
database_embeddings, filenames, database_tree, cosine_distances = load_embeddings_from_database(model)

@app.route('/compare', methods=['POST'])
def compare_faces():
    total_start_time = time.time()

    # Get the photo to compare from the request
    photo: FileStorage = request.files['photo']

    # Convert the photo to an image
    image = Image.open(photo.stream)
    image = np.array(image)

    # Detect faces in the image
    detector_start_time = time.time()
    detector = get_face_detector()
    faces = detector(image, 1)
    detector_end_time = time.time()

    matches = []
    for i, face in enumerate(faces):
        # Extract the region of interest from the image
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # Check that the coordinates are within the bounds of the image
        if x >= 0 and y >= 0 and x + w <= image.shape[1] and y + h <= image.shape[0]:
            roi = image[y:y+h, x:x+w]

            # Preprocess the face and compute its embedding
            inference_start_time = time.time()
            embedding = model.compute_embedding(roi)
            inference_end_time = time.time()

            # Compare the embedding to the faces in the database
            db_search_start_time = time.time()
            match = compare_to_database_cosine(embedding, database_embeddings, filenames)
            db_search_end_time = time.time()

            matches.append({
                'face': i, 
                'match': match, 
                'inference_time': inference_end_time - inference_start_time,
                'db_search_time': db_search_end_time - db_search_start_time
            })

    detector_time = detector_end_time - detector_start_time
    total_end_time = time.time()
    total_time = total_end_time - total_start_time

    # Return the result as JSON
    return jsonify({
        'matches': matches,
        'detector_time': detector_time,
        'total_time': total_time
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)