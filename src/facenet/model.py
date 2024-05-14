import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1
import torch

class FaceNetModel:
    def __init__(self, pretrained_model="vggface2"):
        self.pretrained_model = pretrained_model
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_model(self):
        self.model = InceptionResnetV1(pretrained=self.pretrained_model).eval()
        self.model = self.model.to(self.device)
        print(self.model)

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

        # Convert the image to a PyTorch tensor
        preprocessed_image = torch.from_numpy(preprocessed_image)

        # Transpose the image to the format expected by PyTorch
        preprocessed_image = preprocessed_image.permute(2, 0, 1)

        # Add a batch dimension
        preprocessed_image = preprocessed_image.unsqueeze(0)

        # Make sure the image is on the same device as the model
        preprocessed_image = preprocessed_image.to(self.device)

        # Compute the embedding
        embedding = self.model(preprocessed_image)

        # Convert the embedding to a numpy array
        embedding = embedding.detach().cpu().numpy()

        return embedding

    def compare_faces(self, image, database, threshold=0.5):
        query_embedding = self.compute_embedding(image)
        for face_image in database:
            face_embedding = self.compute_embedding(face_image)
            distance = np.linalg.norm(query_embedding - face_embedding)
            if distance < threshold:
                return True
        return False
