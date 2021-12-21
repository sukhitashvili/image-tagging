import cv2
import numpy as np
import streamlit as st
from PIL import Image


@st.cache(allow_output_mutation=True)
def get_predictor_model():
    from model import Model
    model = Model()
    return model


header = st.container()
model = get_predictor_model()

with header:
    st.title('Hello!')
    st.text(
        'tag images with the most descriptive text labels from a predefined set of the text labels')

uploaded_file = st.file_uploader("Or choose an image...")
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image = np.array(image)
    prediction = model.predict(image=image)
    st.write(f'Predicted labels:')
    for lbl, conf in zip(prediction['labels'], prediction['confidences']):
        st.write(f'\t\tlabel: **{lbl}**  with confidence of: **{np.round(conf, 3)}**')
    st.write('Original Image')
    if len(image.shape) == 3:
        cv_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image)
