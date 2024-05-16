import os
import cv2
import numpy as np
import pickle
from scipy.spatial import KDTree
from sklearn.metrics.pairwise import cosine_distances
import dlib
# Initialize the face detector
import dlib

def get_face_detector():
    # Initialize the face detector
    detector = dlib.get_frontal_face_detector()
    return detector


def load_embeddings_from_database(model):
    database_path = os.path.join(os.path.dirname(__file__), '../db/faces')
    embeddings_path = os.path.join(os.path.dirname(__file__), '../db/embeddings.pkl')
    cosine_distances_path = os.path.join(os.path.dirname(__file__), '../db/cosine_distances.pkl')
    
    if os.path.exists(embeddings_path):
        with open(embeddings_path, 'rb') as f:
            database_embeddings, filenames, database_tree = pickle.load(f)
    else:
        database_embeddings = []
        filenames = []
        detector = get_face_detector()
        for filename in os.listdir(database_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(database_path, filename)
                image = cv2.imread(image_path)
                
                # Detect faces in the image
                faces = detector(image, 1)

                for i, face in enumerate(faces):
                    # Extract the region of interest from the image
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()

                    # Check that the coordinates are within the bounds of the image
                    if x >= 0 and y >= 0 and x + w <= image.shape[1] and y + h <= image.shape[0]:
                        roi = image[y:y+h, x:x+w]

                        embedding = model.compute_embedding(roi)
                        embedding = np.reshape(embedding, (1, -1))  # Reshape the embedding into a 2D array
                        database_embeddings.append(embedding)
                        filenames.append(f"{filename}_face{i}")

        database_embeddings = np.concatenate(database_embeddings, axis=0)  # Concatenate the embeddings into a 2D array
        database_tree = KDTree(database_embeddings)

        with open(embeddings_path, 'wb') as f:
            pickle.dump((database_embeddings, filenames, database_tree), f)

    if os.path.exists(cosine_distances_path):
        with open(cosine_distances_path, 'rb') as f:
            cosine_dist = pickle.load(f)
    else:
        cosine_dist = cosine_distances(database_embeddings)
        with open(cosine_distances_path, 'wb') as f:
            pickle.dump(cosine_dist, f)

    return database_embeddings, filenames, database_tree, cosine_dist

def calculate_euclidean_distance(embedding1, embedding2):
    return np.sqrt(np.sum((embedding1 - embedding2) ** 2))

def calculate_cosine_distance(embedding1, embedding2):
        embedding1 = np.reshape(embedding1, (1, -1))  # Reshape the embedding into a 2D array
        embedding2 = np.reshape(embedding2, (1, -1))  # Reshape the embedding into a 2D array
        distance = cosine_distances(embedding1, embedding2)
        return distance[0][0]

def compare_to_database(embedding, database_embeddings, filenames, database_tree):
    distances, indices = database_tree.query(embedding, k=len(database_embeddings))
    min_distance = np.min(distances)
    min_index = np.argmin(distances)
    sorted_indices = np.argsort(distances)
    print("Filenames sorted according to distances:")
    for i in sorted_indices[0]:
        print(f"{filenames[indices[0][i]]}: {distances[0][i]}")
    if min_distance < 0.68:  # This threshold may need to be adjusted based on your specific use case
        return filenames[indices[0][min_index]]

    return None

from sklearn.metrics.pairwise import cosine_similarity

def compare_to_database_cosine(embedding, database_embeddings, filenames, n=1):
    # Calculate cosine distances
    distances = 1 - cosine_similarity(embedding.reshape(1, -1), database_embeddings)
    
    sorted_indices = np.argsort(distances)
    print("Filenames sorted according to distances:")
    for i in sorted_indices[0]:
        print(f"{filenames[i]}: {distances[0][i]}")
    
    # Get the first 'n' matches
    first_n_matches = [filenames[i] for i in sorted_indices[0][:n]]
    
    return first_n_matches