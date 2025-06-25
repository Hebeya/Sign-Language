import streamlit as st
import tensorflow as tf
from PIL import Image as PILImage
import numpy as np
import os
import random

# Load model
loaded_CNN = tf.keras.models.load_model('Two_way_Communication_system_model.h5')

# Class names
class_names = ['again', 'agree', 'answer', 'attendance', 'book', 'break', 'careful', 'change',
               'chat', 'congratulations', 'email', 'file', 'good morning', 'happy birthday',
               'home', 'how are you', 'hungry', 'i need help', 'join', 'keepsmile', 'meet', 
               'mistake', 'open', 'opinion', 'pass', 'please', 'practice', 'pressure', 'problem',
               'questions', 'remember', 'seat', 'shift', 'sick', 'stop', 'sun', 'team', 'thirsty', 
               'this', 'together', 'understand', 'wait', 'where', 'write']

# Folder containing phrase folders
PHRASES_FOLDER = "images for phrases"  # update this path if needed

st.title("Phrase → Image → Prediction")

phrase = st.text_input("Enter a phrase:", "")

if phrase:
    phrase = phrase.strip().lower()
    if phrase in class_names:
        folder_path = os.path.join(PHRASES_FOLDER, phrase)
        if os.path.exists(folder_path):
            images = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if images:
                chosen_image_file = random.choice(images)
                image_path = os.path.join(folder_path, chosen_image_file)

                # Display the chosen image
                st.image(PILImage.open(image_path), caption=f"Image from '{phrase}' folder", use_container_width=True)

                # Prepare image for prediction
                img = PILImage.open(image_path).resize((256, 256)) # Change to your model's input size
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Predict
                predictions = loaded_CNN.predict(img_array)
                predicted_class_idx = np.argmax(predictions, axis=1)[0]
                predicted_class_name = class_names[predicted_class_idx]

                # Show result
                st.success(f"Predicted phrase: **{predicted_class_name}**")
            else:
                st.warning(f"No images found in '{phrase}' folder.")
        else:
            st.error(f"Folder '{phrase}' not found in '{PHRASES_FOLDER}'.")
    else:
        st.warning("Invalid phrase. Please enter a valid phrase from the class list.")
