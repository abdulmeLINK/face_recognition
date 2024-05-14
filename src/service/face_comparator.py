import os
import cv2
import numpy as np

import os
import cv2
import numpy as np
import pickle

def load_embeddings_from_database(model):
    # Define the path to the database
    database_path = os.path.join(os.path.dirname(__file__), '../db/faces')

    # Define the path to the saved embeddings
    embeddings_path = os.path.join(os.path.dirname(__file__), '../db/embeddings.pkl')

    # If the embeddings have already been computed and saved, load them
    if os.path.exists(embeddings_path):
        with open(embeddings_path, 'rb') as f:
            database_embeddings = pickle.load(f)
    else:
        # Otherwise, compute the embeddings
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

        # Save the embeddings for future use
        with open(embeddings_path, 'wb') as f:
            pickle.dump(database_embeddings, f)

    return database_embeddings

def calculate_euclidean_distance(embedding1, embedding2):
    return np.sqrt(np.sum((embedding1 - embedding2) ** 2))

def compare_to_database(embedding, database_embeddings):
    database_embeddings = load_embeddings_from_database()

    for db_embedding in database_embeddings:
        distance = calculate_euclidean_distance(embedding, db_embedding)
        if distance < 0.5:  # This threshold may need to be adjusted based on your specific use case
            return True

    return False