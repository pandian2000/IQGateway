import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the model architecture from the JSON file
with open('malaria_detector_architecture.json', 'r') as f:
    loaded_model_json = f.read()

# Reconstruct the model from the JSON file
malaria_detector = tf.keras.models.model_from_json(loaded_model_json)

# Load the model weights
malaria_detector.load_weights('malaria_detector_weights.weights.h5')

# Function to preprocess image for classification and highlight infected regions
def preprocess_and_highlight(image):
    # Preprocess the image (resize, normalize, etc.)
    resized_image = cv2.resize(image, (50, 50))
    normalized_image = resized_image / 255.0  # Normalize pixel values
    
    # Perform classification using the loaded model
    prediction = malaria_detector.predict(np.expand_dims(normalized_image, axis=0))
    
    # Extract the predicted class (assuming binary classification)
    predicted_class = np.argmax(prediction)
    
    # Highlight infected regions
    if predicted_class == 1:
        # Convert image to grayscale
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
        
        # Apply thresholding to segment dark regions (potential infected areas)
        _, thresh = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours of potential infected areas
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on the original image to highlight infected areas
        highlighted_image = resized_image.copy()
        cv2.drawContours(highlighted_image, contours, -1, (255, 0, 0), 2)  # Draw contours in red
        
        # Determine if the cell is infected and label it accordingly
        label = 'Infected'
    else:
        # No infection detected, return the original image
        highlighted_image = resized_image
        
        # Label the cell as uninfected
        label = 'Uninfected'
    
    return highlighted_image, label

# Streamlit interface
st.title('Malaria Cell Image Classifier')

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read uploaded image
    image = np.array(Image.open(uploaded_file))
    
    # Preprocess and highlight infected regions
    highlighted_image, label = preprocess_and_highlight(image)
    
    # Display the classified image with highlighted infected regions and label
    st.image(highlighted_image, caption=f'Classification: {label}', use_column_width=True)
