import streamlit as st
from PIL import Image

# Data
from pandas import read_csv

# Model
from keras.models import load_model
import tensorflow as tf

# Button callback function
def predict_neighboring_image(offset):
    st.session_state.image_index = st.session_state.image_index + offset

def main():
    st.success("# COVID-19 CT scan classification")

    # uploaded_file=st.file_uploader("Upload your scan here...",type=['jpg','png'])

    if 'image_index' not in st.session_state:
        # Initialize image to use at session start
        st.session_state.image_index = 0

    # Load image and model
    print("Loading dataset...")
    dataset = read_csv('original.csv', nrows=1, skiprows=st.session_state.image_index)
    # Remove class label
    dataset = dataset.iloc[:, :-1]
    image_pixels = dataset.iloc[0]

    print("Loading model...")
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
    predictions = model.predict(dataset)
    if predictions[0] == 1:
        prediction = 'Positive'
    else:
        prediction = 'Negative'

    "Image ", st.session_state.image_index

    st.image(image, caption=prediction, use_column_width=True)

    st.sidebar.button('Predict Previous Scan', on_click=predict_neighboring_image, kwargs={'offset': -1})
    st.sidebar.button('Predict Next Scan', on_click=predict_neighboring_image, kwargs={'offset': 1})

    # if uploaded_file is not None:
    #     image = Image.open(uploaded_file)
    #     st.image(image, caption='Uploaded CT scan', use_column_width=True)

if __name__=='__main__':
    main()