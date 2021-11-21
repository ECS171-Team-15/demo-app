import streamlit as st
from PIL import Image

def main ():
    """ All code goes here  """
    st.success("# COVID-19 CT scan classification")
    uploaded_file=st.file_uploader("Upload your scan here...",type=['jpg','png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded CT scan', use_column_width=True)

if __name__=='__main__':
    main()