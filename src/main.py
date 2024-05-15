import argparse
import cv2
from facenet.model import FaceNetModel
from service.face_comparator import load_embeddings_from_database, compare_to_database, calculate_cosine_distance, compare_to_database_cosine

# Load the FaceNet model
model = FaceNetModel()
model.load_model()

# Load the embeddings from the database
database_embeddings, filenames, database_tree, cosine_distances = load_embeddings_from_database(model)

# Process the input photo and compare it to the embeddings in the database
def compare_faces(input_photo, use_cosine_distance=False):
    input_face = cv2.imread(input_photo)
    input_embedding = model.compute_embedding(input_face)
    if use_cosine_distance:
        result = compare_to_database_cosine(input_embedding, database_embeddings, filenames, cosine_distances)
    else:
        result = compare_to_database(input_embedding, database_embeddings, filenames, database_tree)
    return result

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Compare faces.')
parser.add_argument('input_photo', type=str, help='Path to the input photo.')
parser.add_argument('--pairwise', type=str, help='Path to the second photo for pairwise comparison.')
parser.add_argument('--cosine_db', action='store_true', help='If set, compare the input photo to the database using cosine distance.')
args = parser.parse_args()

# Example usage
if args.pairwise:
    photo1 = cv2.imread(args.input_photo)
    photo2 = cv2.imread(args.pairwise)
    embedding1 = model.compute_embedding(photo1)
    embedding2 = model.compute_embedding(photo2)
    distance = calculate_cosine_distance(embedding1, embedding2)
    print(f'Distance between input photo and pairwise photo: {distance}')
else:
    result = compare_faces(args.input_photo, args.cosine_db)
    print(result)