import os
from pipeline import preprocess_image, generate_embedding, train_classifier
import numpy as np

# Define directories
input_dir = "."  # Base directory with folders for each person
output_dir = "cropped_faces"               # Directory to store cropped faces
os.makedirs(output_dir, exist_ok=True)

# Step 1: Preprocess Images
embeddings = []
labels = []

print("Preprocessing images and generating embeddings...")
for person in os.listdir(input_dir):
    person_path = os.path.join(input_dir, person)
    if os.path.isdir(person_path):  # Ensure it's a directory
        person_output_dir = os.path.join(output_dir, person)
        os.makedirs(person_output_dir, exist_ok=True)

        for photo in os.listdir(person_path):
            photo_path = os.path.join(person_path, photo)
            output_photo_path = os.path.join(person_output_dir, photo)

            # Skip non-image files
            if not photo.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            try:
                preprocess_image(photo_path, output_photo_path)
                print(f"Processed: {photo_path}")

                embedding = generate_embedding(output_photo_path)
                embeddings.append(embedding)
                labels.append(person)

            except Exception as e:
                print(f"Error processing {photo_path}: {e}")

# Step 2: Train the Classifier
if len(embeddings) > 0:
    embeddings = np.array(embeddings)
    print("Training classifier...")
    train_classifier(embeddings, labels, "classifier.pkl")
    print("Classifier saved as 'classifier.pkl'")
else:
    print("No embeddings were generated. Check your input images.")
