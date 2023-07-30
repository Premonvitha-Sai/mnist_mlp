import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import tensorflow
import keras

def preprocess(image):
    image = image.resize((28,28))
    image = image.convert('L')
    image = np.array(image)
    image = 255 - image  # Invert colors
    image = image.reshape(1, 784).astype('float32')
    image = image / 255
    return image

def main():
    st.title("MNIST Digit Classification with MLP")
    model = load_model('model.h5', compile=False)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    st.subheader("Draw a digit below:")
    canvas_result = st_canvas(
        fill_color="#ffffff",  # Fixed fill color with some opacity
        stroke_width=20,
        stroke_color='#000000',
        background_color="#ffffff",
        width=280,
        height=280,
        drawing_mode='freedraw',
        key="canvas",
    )

    if st.button('Predict'):
        if canvas_result.image_data is not None:
            img = Image.fromarray((canvas_result.image_data).astype(np.uint8))
            if np.max(img) > 0:  # If the image is not empty (i.e., the user drew something on the canvas)
                preprocessed = preprocess(img)
                prediction = np.argmax(model.predict(preprocessed), axis=-1)
                st.write(f'Predicted digit is: {prediction[0]}')
            else:
                st.write("Please draw a digit.")
        else:
            st.write("Please draw a digit.")

if __name__ == "__main__":
    main()
