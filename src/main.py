import argparse
import cv2
from facenet.model import FaceNetModel
from service.face_comparator import get_face_detector, load_embeddings_from_database, compare_to_database, calculate_cosine_distance, compare_to_database_cosine

# Load the FaceNet model
model = FaceNetModel()
model.load_model()

# Load the embeddings from the database
database_embeddings, filenames, database_tree, cosine_distances = load_embeddings_from_database(model)

# Process the input photo and compare it to the embeddings in the database
def compare_faces(input_photo, use_cosine_distance=False):
    input_image = cv2.imread(input_photo)

    # Detect faces in the image
    detector = get_face_detector()
    faces = detector(input_image, 1)

    results = []
    for i, face in enumerate(faces):
        # Extract the region of interest from the image
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # Check that the coordinates are within the bounds of the image
        if x >= 0 and y >= 0 and x + w <= input_image.shape[1] and y + h <= input_image.shape[0]:
            roi = input_image[y:y+h, x:x+w]

            # Compute the embedding for the face
            input_embedding = model.compute_embedding(roi)

            # Compare the embedding to the embeddings in the database
            if use_cosine_distance:
                result = compare_to_database_cosine(input_embedding, database_embeddings, filenames)
            else:
                result = compare_to_database(input_embedding, database_embeddings, filenames, database_tree)
            
            results.append({'face': i, 'match': result})

    return results


def pairwise(input_photo1, input_photo2, use_cosine_distance=False):
    input_image1 = cv2.imread(input_photo1)
    input_image2 = cv2.imread(input_photo2)

    # Detect faces in the image
    detector = get_face_detector()
    faces1 = detector(input_image1, 1)
    faces2 = detector(input_image2, 1)

    results = []
    for i, face1 in enumerate(faces1):
        # Extract the region of interest from the image
        x1, y1, w1, h1 = face1.left(), face1.top(), face1.width(), face1.height()

        # Check that the coordinates are within the bounds of the image
        if x1 >= 0 and y1 >= 0 and x1 + w1 <= input_image1.shape[1] and y1 + h1 <= input_image1.shape[0]:
            roi1 = input_image1[y1:y1+h1, x1:x1+w1]

            # Compute the embedding for the face
            input_embedding1 = model.compute_embedding(roi1)

            for j, face2 in enumerate(faces2):
                # Extract the region of interest from the image
                x2, y2, w2, h2 = face2.left(), face2.top(), face2.width(), face2.height()

                # Check that the coordinates are within the bounds of the image
                if x2 >= 0 and y2 >= 0 and x2 + w2 <= input_image2.shape[1] and y2 + h2 <= input_image2.shape[0]:
                    roi2 = input_image2[y2:y2+h2, x2:x2+w2]

                    # Compute the embedding for the face
                    input_embedding2 = model.compute_embedding(roi2)

                    # Compare the embeddings
                    if use_cosine_distance:
                        result = cosine_distance(input_embedding1, input_embedding2)
                    else:
                        result = euclidean_distance(input_embedding1, input_embedding2)

                    results.append({'face1': i, 'face2': j, 'match': result})

    return results

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Compare faces.')
parser.add_argument('input_photo', type=str, help='Path to the input photo.')
parser.add_argument('--pairwise', type=str, help='Path to the second photo for pairwise comparison.')
parser.add_argument('--cosine_db', action='store_true', help='If set, compare the input photo to the database using cosine distance.')
args = parser.parse_args()

# Example usage
if args.pairwise:
    results = pairwise(args.input_photo, args.pairwise, args.cosine_db)
    print(f'Distance between input photo and pairwise photo: {distance}')
else:
    result = compare_faces(args.input_photo, args.cosine_db)
    print(result)