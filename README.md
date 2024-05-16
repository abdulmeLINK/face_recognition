# Comparison Service

This project provides a face comparison service using the VGGface2 model. It includes a REST API for comparing a given photo to a database of faces.

## Project Structure

- `requirements.txt`: This file lists the Python dependencies required for the project. It includes the necessary packages for running the VGGface2 model and setting up the REST API.
- `src/`: This directory contains the source code for the project.
- `src/db/faces/`: This directory is where you should place the images you want to compare.

## Usage

To use the face comparison service, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/face_recognition.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Place the images you want to compare in the `src/db/faces` directory.

4. Start the service:

   ```bash
   python api.py
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
   docker build -t face_recognition .
   ```

2. Run the Docker container:

   ```bash
   docker run -p 5000:5000 face_recognition
   ```

   The service will start running inside the Docker container and will be accessible at `http://localhost:5000`.

## Notes

- Make sure you have a compatible GPU and the necessary drivers installed to take advantage of GPU acceleration.

