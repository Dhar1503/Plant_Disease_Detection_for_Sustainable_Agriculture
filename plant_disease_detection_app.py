import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('C:/Users/user/Edunet/plant_disease_model.h5')


class_names = [
                'Apple__Apple_scab', 'Apple__Black_rot', 'Apple__Cedar_apple_rust', 'Apple__healthy',
                'Blueberry__healthy', 'Cherry__Powdery_mildew', 'Cherry__healthy',
                'Corn__Cercospora_leaf_spot', 'Corn__Common_rust', 'Corn__Northern_Leaf_Blight',
                'Corn__healthy', 'Grape__Black_rot', 'Grape__Esca_(Black_Measles)',
                'Grape__Leaf_blight', 'Grape__healthy', 'Orange__Citrus_greening',
                'Peach__Bacterial_spot', 'Peach__healthy', 'Pepper__Bacterial_spot',
                'Pepper__healthy', 'Potato__Early_blight', 'Potato__Late_blight', 'Potato__healthy',
                'Raspberry__healthy', 'Soybean__healthy', 'Squash__Powdery_mildew',
                'Strawberry__Leaf_scorch', 'Strawberry__healthy', 'Tomato__Bacterial_spot',
                'Tomato__Early_blight', 'Tomato__Late_blight', 'Tomato__Leaf_Mold',
                'Tomato__Septoria_leaf_spot', 'Tomato__Spider_mites', 'Tomato__Target_Spot',
                'Tomato__Yellow_Leaf_Curl_Virus', 'Tomato__Mosaic_virus', 'Tomato__healthy'
            ]

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((150, 150))  # Resize image to match the input shape of the model
    img = np.array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Streamlit App Layout
st.title("Plant Disease Detection")
st.write("Upload an image of a plant leaf to check for disease.")

# Upload image
uploaded_img = st.file_uploader("Choose a plant image", type=["jpg", "png", "jpeg"])

if uploaded_img is not None:
    # Display the uploaded image
    img = image.load_img(uploaded_img, target_size=(150, 150))
    st.image(img, caption="Uploaded Image.", use_column_width=True)

    # Preprocess the image for prediction
    img_array = preprocess_image(img)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    # Display prediction result
    st.write(f"Prediction: {predicted_class}")
