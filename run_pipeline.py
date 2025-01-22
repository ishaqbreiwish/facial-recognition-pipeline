from pipeline import preprocess_image, generate_embedding, predict_face
import os

# Step 1: Preprocess the Test Image
test_image = "path_to_test_image.jpg"  # Replace with the path to your test image
cropped_image = "cropped_test_image.jpg"

try:
    preprocess_image(test_image, cropped_image)
    print(f"Preprocessed: {test_image}")

    # Step 2: Generate Embedding
    embedding = generate_embedding(cropped_image)
    print(f"Generated Embedding: {embedding}")

    # Step 3: Predict the Face
    result, probabilities = predict_face(embedding, "classifier.pkl")
    print(f"Predicted: {result}")
    print(f"Probabilities: {probabilities}")

except Exception as ex:
    print(f"Error during pipeline execution: {e}")
