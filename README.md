# Presentation Demo

## Prerequisites

1. One of our four model files (\*.h5)
2. Streamlit library
3. Tensorflow and Keras versions >=2.6.0
4. pip3 install -U protobuf1

## Setup

1. Install streamlit by running: `python3 -m pip install streamlit`
2. Ensure you have the proper version of Tensorflow and Keras
3. Download the best-performing model in this Google Drive [folder](https://drive.google.com/drive/folders/1lgG4LkhwK06ysk9o09jS8ABqopvbBQYz). (According to our paper, the best model is the one that only has L1_L2 regularization, or `Falsel1_l2.h5`)

## Running

1. Run `python3 -m streamlit run app.py` to run the web server locally on your computer.
2. Select a CT-scan from our dataset (or anywhere else)
3. Press `Ctrl+C` to stop the web server.
