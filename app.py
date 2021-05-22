import streamlit as st
import pandas as pd
import cv2
import numpy as np
from utils import word_puzzle
from PIL import Image

try:
    col1, col2, col3 = st.beta_columns([4, 12, 4])

    with col2:
        st.markdown('# **WORD PUZZLE SOLVER**')

    st.text('')
    st.text('')
    st.text('')

    image_file = st.file_uploader(
        "Upload An Image", type=["jpg", "jpeg", "png"])

    col4, col5, col6 = st.beta_columns([5, 10, 5])

    op = None

    if image_file is not None:
        image = Image.open(image_file).convert('RGB')
        image.save('../Images/input.jpg')
        with col5:
            st.markdown('### **INPUT IMAGE**')
            st.image(image_file, caption='Input Image')
        col7, col8 = st.beta_columns([5, 5])
        with col7:
            nrows = st.number_input('Number Of Rows', 0)
        with col8:
            ncols = st.number_input('Number Of Columns', 0)
        words = st.text_input('Enter Words To Search', '')
        words = ' '.join(words.split(','))
        words = ' '.join(words.split('\n'))
        words = words.split()
        col9, col10, col11 = st.beta_columns([22, 6, 22])
        with col10:
            process_img = st.button("Solve")
        if process_img:
            if nrows < 2:
                st.error('Number of rows must be atleast 2')
            elif ncols < 2:
                st.error('Number of columns must be atleast 2')
            elif len(words) < 1:
                st.error('Enter words to be searched')
            else:
                st.success('Processing')
                op = word_puzzle('../Images/input.jpg', nrows, ncols, words)
                st.success('Done')
        col12, col13, col14 = st.beta_columns([5, 10, 5])
        if op is not None:
            with col13:
                st.text('')
                st.text('')
                st.markdown('### **OUTPUT IMAGE**')
                st.image(op, caption='Output Image')
                st.balloons()

except:
    pass
