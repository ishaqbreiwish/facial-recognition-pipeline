# Facial Recognition Pipeline Code

## 1. Preprocessing with Dlib
# Detect, align, and crop faces
from dlib import get_frontal_face_detector, shape_predictor
from PIL import Image
import numpy as np

# Load Dlib tools
detector = get_frontal_face_detector()
shape_predictor = shape_predictor("shape_predictor_68_face_landmarks.dat")

def preprocess_image(image_path, output_path):
    image = Image.open(image_path)
    image_array = np.array(image)
    faces = detector(image_array)

    if len(faces) > 0:
        # Assume largest face is the target
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        shape = shape_predictor(image_array, face)
        landmarks = [(p.x, p.y) for p in shape.parts()]

        # Crop around the face based on landmarks (e.g., eyes and mouth)
        cropped_image = image.crop((face.left(), face.top(), face.right(), face.bottom()))
        cropped_image.save(output_path)
        return output_path
    else:
        raise ValueError("No face detected")

## 2. Creating Embeddings with TensorFlow
# Generate embeddings using a pre-trained model (e.g., Inception ResNet V1)
import tensorflow as tf
from scipy.spatial.distance import cosine

# Load the pre-trained model
model_path = "20170511-185253.pb"
def load_model():
    with tf.io.gfile.GFile(model_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")

load_model()

def generate_embedding(image_path):
    with tf.compat.v1.Session() as sess:
        input_tensor = sess.graph.get_tensor_by_name("input:0")
        embeddings_tensor = sess.graph.get_tensor_by_name("embeddings:0")

        # Preprocess image to match model requirements
        image = Image.open(image_path).resize((160, 160))
        image_array = np.array(image) / 255.0

        # Run the model
        feed_dict = {input_tensor: [image_array]}
        embeddings = sess.run(embeddings_tensor, feed_dict=feed_dict)
        return embeddings[0]

## 3. Training the SVM Classifier
from sklearn.svm import SVC
import joblib

# Train the classifier
def train_classifier(embeddings, labels, output_path):
    classifier = SVC(kernel="linear", probability=True)
    classifier.fit(embeddings, labels)
    joblib.dump(classifier, output_path)

# Predict using the classifier
def predict_face(embedding, classifier_path):
    classifier = joblib.load(classifier_path)
    probabilities = classifier.predict_proba([embedding])
    predicted_label = classifier.classes_[np.argmax(probabilities)]
    return predicted_label, probabilities

# Example usage (terminal commands provided below)
# 1. Preprocess an image and save the cropped face
# preprocess_image("example.jpg", "cropped_face.jpg")

# 2. Generate embeddings for faces
# embedding = generate_embedding("cropped_face.jpg")

# 3. Train the classifier with a set of embeddings and labels
# embeddings = [embedding1, embedding2, ...]  # List of embeddings
# labels = ["Alice", "Bob", ...]  # Corresponding labels
# train_classifier(embeddings, labels, "classifier.pkl")

# 4. Predict a face
# result = predict_face(new_embedding, "classifier.pkl")
