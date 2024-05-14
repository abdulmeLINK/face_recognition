# FaceNet Comparison

This project implements FaceNet512, a face recognition model, to compare faces to a database of images. If a match is found, it indicates that the given photo matches one of the images in the database. The project is implemented end-to-end with high performance and utilizes the GPU if available. It is also dockerized for easy deployment.

## Project Structure

```
facenet-comparison
├── src
│   ├── main.py
│   ├── facenet
│   │   └── model.py
│   ├── service
│   │   └── api.py
│   ├── utils
│   │   └── gpu_utils.py
│   └── db
│       └── faces
│           └── .gitkeep
├── Dockerfile
├── requirements.txt
└── README.md
```

The project consists of the following files:

- `src/main.py`: This file serves as the entry point of the application. It contains the main logic for comparing faces using the FaceNet512 model. It loads the model, processes the input photo, and compares it to the faces in the database.

- `src/facenet/model.py`: This file exports the `FaceNetModel` class, which encapsulates the FaceNet512 model. It provides methods for loading the model, preprocessing images, and computing face embeddings.

- `src/service/api.py`: This file exports the `FaceComparisonService` class, which sets up a REST API for calling the face comparison functionality. It provides an endpoint for comparing a given photo to the faces in the database.

- `src/utils/gpu_utils.py`: This file exports utility functions for checking the availability of a GPU and configuring TensorFlow to use the GPU if available.

- `src/db/faces/.gitkeep`: This file is a placeholder file to ensure that the `faces` directory is included in version control.

- `Dockerfile`: This file is used to build a Docker image for the project. It specifies the base image, installs the necessary dependencies, and sets up the project files.

- `requirements.txt`: This file lists the Python dependencies required for the project. It includes the necessary packages for running the FaceNet512 model and setting up the REST API.

## Usage

To use the face comparison service, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/facenet-comparison.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Place the images you want to compare in the `src/db/faces` directory.

4. Start the service:

   ```bash
   python src/main.py
   ```

5. The service will start running and listening for requests. You can now call the API endpoint to compare a photo to the faces in the database.

   Example API request:

   ```bash
   POST /compare
   Content-Type: multipart/form-data

   photo=<photo-file>
   ```

   The response will indicate whether a match was found or not.

## Docker

To run the project using Docker, follow these steps:

1. Build the Docker image:

   ```bash
   docker build -t facenet-comparison .
   ```

2. Run the Docker container:

   ```bash
   docker run -p 5000:5000 facenet-comparison
   ```

   The service will start running inside the Docker container and will be accessible at `http://localhost:5000`.

## Notes

- Make sure you have a compatible GPU and the necessary drivers installed to take advantage of GPU acceleration.

- The project assumes that the FaceNet512 model and the necessary database files are already available. You will need to provide these files or implement the functionality to load them.

- This README file is intentionally left blank.