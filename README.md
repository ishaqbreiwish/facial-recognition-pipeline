# Facial Recognition Pipeline

This project demonstrates a facial recognition pipeline using pre-trained TensorFlow models, Dlib for face detection and alignment, and a Support Vector Machine (SVM) classifier for face recognition. The pipeline supports batch processing for training and individual execution for testing new images.

---

## Features

- **Face Preprocessing**: Detects and aligns faces using Dlib.
- **Embedding Generation**: Converts faces into 128-dimensional embeddings using a pre-trained TensorFlow model.
- **Classifier Training**: Trains an SVM classifier on embeddings for face recognition.
- **Batch Processing**: Automates preprocessing and training for multiple images.
- **Pipeline Execution**: Supports testing individual images to predict the associated person.

---

## Folder Structure

```
facial_recognition_pipeline/
├── batch_pipeline.py                # Automates preprocessing and training
├── run_pipeline.py                  # Runs the pipeline for a test image
├── pipeline.py                      # Core pipeline functions
├── shape_predictor_68_face_landmarks.dat  # Dlib face landmark model
├── 20170511-185253.pb               # Pre-trained TensorFlow model
├── leo/                             # Images for "Leo"
├── ryan/                            # Images for "Ryan"
├── cropped_faces/                   # Directory for cropped faces (auto-created)
```

---

## Setup

### 1. Install Dependencies

Activate your virtual environment and install the required packages:

```bash
pip install tensorflow dlib scikit-learn joblib numpy pillow
```

### 2. Download Pre-Trained Models

- **Dlib Model**: [Download here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

  - Extract the `.bz2` file:
    ```bash
    bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
    ```

- **TensorFlow Model**: [Download here](https://github.com/davidsandberg/facenet/raw/master/src/models/20170511-185253/20170511-185253.pb)

Place these files in the project directory.

---

## Usage

### 1. Batch Processing

Use `batch_pipeline.py` to preprocess images and train the classifier:

```bash
python3 batch_pipeline.py
```

- **Input**: Place your images in subfolders named after their respective classes (e.g., `leo/` and `ryan/`).
- **Output**: Cropped faces are saved in `cropped_faces/`, and the trained classifier is saved as `classifier.pkl`.

### 2. Test the Pipeline

Use `run_pipeline.py` to test the pipeline with a new image:

```bash
python3 run_pipeline.py
```

- Update the `test_image` variable in the script to the path of your test image.
- **Output**: Predicted class (e.g., `leo` or `ryan`) and probabilities are displayed.

### 3. Manual Execution

#### Preprocess a Single Image:

```bash
python3 -c "from pipeline import preprocess_image; preprocess_image('input.jpg', 'cropped.jpg')"
```

#### Generate Embedding for a Face:

```bash
python3 -c "from pipeline import generate_embedding; print(generate_embedding('cropped.jpg'))"
```

#### Predict a Face:

```bash
python3 -c "from pipeline import predict_face, generate_embedding; embedding = generate_embedding('test.jpg'); result, probabilities = predict_face(embedding, 'classifier.pkl'); print('Predicted:', result); print('Probabilities:', probabilities)"
```

---

## Notes

1. Ensure all image files are in `.jpg`, `.jpeg`, or `.png` format.
2. Avoid using non-image files like `.DS_Store` or `.pyc` files in your input directories.
3. Test images should not be part of the training data for accurate evaluation.
