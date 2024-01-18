import tensorflow as tf 
import tensorflow_hub as hub 
import streamlit as st 
import numpy as np 
from PIL import Image
from io import BytesIO

st.title("Style your image ")

model = hub.load('https://kaggle.com/models/google/arbitrary-image-stylization-v1/frameworks/TensorFlow1/variations/256/versions/1')

image_style = st.file_uploader("**Import your style image**",type=['png','jpg','jpeg','webp'])
image_content = st.file_uploader("**Import your image**",type=['png','jpg','jpeg','webp'])

if image_content !=None and image_content != None :
    image_style = Image.open(image_style)
    image_style = image_style.convert("RGB")
    image_style = np.array(image_style)
    image_content = Image.open(image_content)
    image_content = image_content.convert("RGB")
    image_content = np.array(image_content)


    content_image = image_content.astype(np.float32)[np.newaxis, ...] / 255.
    style_image = image_style.astype(np.float32)[np.newaxis, ...] / 255.

    button = st.button("Stylize now  !!")
    if button:
        outputs = model(tf.constant(content_image), tf.constant(style_image))
        result = outputs[0]
        numpy_array = np.array(result.numpy()[0] * 255, dtype=np.uint8)

        pillow_image = Image.fromarray(numpy_array)

        output_path = "output_image.jpg"
        pillow_image.save(output_path)
        st.image([output_path], caption='Stylized Image', use_column_width=True)
        st.download_button(
            label="Download Stylized Image",
            data=BytesIO(numpy_array.tobytes()),
            file_name="output_image.jpg",
            key="download_button",
            help="Click to download the stylized image.",
        )