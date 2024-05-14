import tensorflow as tf
import numpy as np
from facenet_pytorch import InceptionResnetV1

class FaceNetModel:
    def __init__(self, pretrained_model="vggface2"):
        self.pretrained_model = pretrained_model
        self.model = None

    def load_model(self):
        self.model = InceptionResnetV1(pretrained=self.pretrained_model).eval()
        self.model.summary()

    def preprocess_image(self, image):
        # Resize the image to the size expected by FaceNet (160x160)
        image = cv2.resize(image, (160, 160))

        # Convert the image to float32
        image = image.astype('float32')

        # Normalize the image to the range [0, 1]
        image /= 255.0

        return image

    def compute_embedding(self, image):
        preprocessed_image = self.preprocess_image(image)
        embedding = self.model.predict(np.expand_dims(preprocessed_image, axis=0))
        return embedding

    def compare_faces(self, image, database, threshold=0.5):
        query_embedding = self.compute_embedding(image)
        for face_image in database:
            face_embedding = self.compute_embedding(face_image)
            distance = np.linalg.norm(query_embedding - face_embedding)
            if distance < threshold:
                return True
        return False
