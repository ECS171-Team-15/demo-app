import streamlit as st
from PIL import Image

# Data
from pandas import read_csv
import random

# Model
from keras.models import load_model
import tensorflow as tf

# Convert prediction to string output
def prediction_str(prediction):
    if prediction >= 0.5:
        return 'Positive'
    else:
        return 'Negative'

def main():
    st.success("# COVID-19 CT scan classification")

    # Load image
    image_file = st.file_uploader(label="Upload your CT-scan here...", type=['jpg', 'png'])

    if image_file is None:
        # No image to classify
        return

    # Preprocess image on the fly
    # Convert to grayscale
    image = Image.open(image_file)
    image = image.convert("L")
    # Resize to mean dimensions
    mean_width = 425
    mean_height = 302
    image = image.resize((mean_width, mean_height))
    # Scale data
    pixels = list(image.getdata())
    pixels = [pixel / 255 for pixel in pixels]

    model = load_model('Falsel1_l2.h5')

    # Reshape into 4D tensor
    # Height goes first because its the number of rows
    pixels = tf.reshape(pixels, (1, mean_height, mean_width, 1))

    # Get prediction
    predictions = model.predict(pixels)
    prediction = predictions[0]

    # Output
    st.image(image, use_column_width=True)
    if prediction_str(prediction) == "Positive":
        st.error("# You are positive for COVID!!!")
    elif prediction_str(prediction) == "Negative":
        st.success("# You are negative for COVID!")

if __name__=='__main__':
    main()
