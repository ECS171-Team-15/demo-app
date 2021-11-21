# Presentation Demo

## Prerequisites

This app relies on:
1. Our preprocessed dataset (original.csv)
2. Our model file (\*.h5). We prefer to use the best-performing model.
3. Streamlit library

## Setup

1. Install streamlit by running: `python3 -m pip install streamlit`
2. Download original.csv to this repository's base directory by running: `bash fetch-original-csv.sh`
3. Download the best-performing model in this Google Drive [folder](https://drive.google.com/drive/folders/1lgG4LkhwK06ysk9o09jS8ABqopvbBQYz).

## Running

1. Run `python3 -m streamlit run app.py` to run the web server locally on your computer.
2. Press `Ctrl+C` to stop the web server.
