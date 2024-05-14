import os
import cv2
import numpy as np
import pickle
from scipy.spatial import KDTree

class ArcFaceModel:
    def __init__(self):
        self.model = model_zoo.get_model('arcface_r100_v1')
        self.model.prepare(ctx_id=-1)  # Use CPU

    def preprocess_image(self, image):
        image = cv2.resize(image, (112, 112))
        image = image.astype('float32')
        image /= 255.0
        return image

    def compute_embedding(self, image):
        preprocessed_image = self.preprocess_image(image)
        embedding = self.model.get_embedding(preprocessed_image)
        return embedding

def load_embeddings_from_database(model):
    database_path = os.path.join(os.path.dirname(__file__), '../db/faces')
    embeddings_path = os.path.join(os.path.dirname(__file__), '../db/embeddings.pkl')

    if os.path.exists(embeddings_path):
        with open(embeddings_path, 'rb') as f:
            database_embeddings, filenames, database_tree = pickle.load(f)
    else:
        database_embeddings = []
        filenames = []

        for filename in os.listdir(database_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(database_path, filename)
                image = cv2.imread(image_path)
                embedding = model.compute_embedding(image)
                embedding = np.reshape(embedding, (1, -1))  # Reshape the embedding into a 2D array
                database_embeddings.append(embedding)
                filenames.append(filename)

        database_embeddings = np.concatenate(database_embeddings, axis=0)  # Concatenate the embeddings into a 2D array
        database_tree = KDTree(database_embeddings)

        with open(embeddings_path, 'wb') as f:
            pickle.dump((database_embeddings, filenames, database_tree), f)

    return database_embeddings, filenames, database_tree

def calculate_euclidean_distance(embedding1, embedding2):
    return np.sqrt(np.sum((embedding1 - embedding2) ** 2))

def compare_to_database(embedding, database_embeddings, filenames, database_tree):
    distances, indices = database_tree.query(embedding, k=len(database_embeddings))
    index = np.argmin(distances)
    if distances[index] < 0.5:  # This threshold may need to be adjusted based on your specific use case
        return filenames[indices[index]]

    return None