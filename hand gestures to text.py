import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp
import streamlit as st

# Load the trained CNN model
loaded_CNN = tf.keras.models.load_model('Two_way_Communication_system_model.h5')

# List of class names
class_names = ['again', 'agree', 'answer', 'attendance', 'book', 'break', 'careful', 'change',
               'chat', 'congratulations', 'email', 'file', 'good morning', 'happy birthday',
               'home', 'how are you', 'hungry', 'i need help', 'join', 'keepsmile', 'meet',
               'mistake', 'open', 'opinion', 'pass', 'please', 'practice', 'pressure', 'problem',
               'questions', 'remember', 'seat', 'shift', 'sick', 'stop', 'sun', 'team', 'thirsty',
               'this', 'together', 'understand', 'wait', 'where', 'write']

st.title("Sign Language Prediction")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Open webcam
if st.button("Start Camera"):
    video = cv2.VideoCapture(0)
    sentence = []
    previous_word = ""
    offset = 20

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        predicted_label = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                x_min = w
                y_min = h
                x_max = 0
                y_max = 0

                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)

                # Add offset
                x_min = max(0, x_min - offset)
                y_min = max(0, y_min - offset)
                x_max = min(w, x_max + offset)
                y_max = min(h, y_max + offset)

                # Draw rectangle
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

                # Crop the hand region
                frameCrop = frame[y_min:y_max, x_min:x_max]
                if frameCrop.size != 0:
                    try:
                        resized = cv2.resize(frameCrop, (256, 256))
                        normalized = resized.astype('float32') / 255.0
                        input_data = np.reshape(normalized, (1, 256, 256, 3))

                        predictions = loaded_CNN.predict(input_data, verbose=0)
                        predicted_index = np.argmax(predictions)
                        predicted_label = class_names[predicted_index]

                        if predicted_label != previous_word:
                            sentence.append(predicted_label)
                            previous_word = predicted_label

                    except Exception as e:
                        print("Resize or prediction failed:", e)

                # Draw hand landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display predictions
        cv2.putText(frame, f'Word: {predicted_label}', (10, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'Sentence: ' + ' '.join(sentence[-10:]), (10, 80),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("Live Feed", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('r'):
            sentence = []
            previous_word = ""

    video.release()
    cv2.destroyAllWindows()
