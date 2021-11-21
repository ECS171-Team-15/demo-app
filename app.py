import streamlit as st
from PIL import Image

# Data
from pandas import read_csv
import random

# Model
from keras.models import load_model
import tensorflow as tf

# Button callback function
def predict_random_ct():
    # 746 images in total for original dataset
    st.session_state.image_index = random.randrange(0, 746)

# Convert label int to string output
def label_str(class_label):
    if class_label == 1:
        return 'Positive'
    else:
        return 'Negative'

def main():
    st.success("# COVID-19 CT scan classification")

    if 'image_index' not in st.session_state:
        # Initialize image to use at session start
        st.session_state.image_index = 0

    # Load image and model
    "Loading image..."
    dataset = read_csv('original.csv', nrows=1, skiprows=st.session_state.image_index)
    # Remove class label
    class_label = dataset.iloc[0, -1]
    dataset = dataset.iloc[:, :-1]

    # Min-max scaler
    "Scaling image pixels..."
    dataset.applymap(lambda x: x / 255)
    image_pixels = dataset.iloc[0]

    "Loading model..."
    model = load_model('Falsel1_l2.h5')

    # Convert image pixels to Image representation
    width = 425
    height = 302
    image = Image.new('L', (width, height))
    image.putdata(image_pixels)

    # Reshape into 4D tensor
    # Height goes first because its the number of rows
    dataset = tf.reshape(dataset, (1, height, width, 1))

    # Model prediction
    "Making prediction..."
    predictions = model.predict(dataset)
    prediction = predictions[0]

    st.image(image, caption="Image " + str(st.session_state.image_index), use_column_width=True)

    "Actual: " + label_str(class_label)
    "Prediction: " + label_str(prediction)

    st.sidebar.button('Predict Random CT Scan', on_click=predict_random_ct)

if __name__=='__main__':
    main()