import streamlit as st
import cv2
import numpy as np
import csv
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Load Models and Classifiers
@st.cache_resource
def load_models():
    model_best = load_model('emotion_detection_model_100epochs.h5')  # Replace with your model path
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    ageNet = cv2.dnn.readNet("age_net.caffemodel", "age_deploy.prototxt")
    genderNet = cv2.dnn.readNet("gender_net.caffemodel", "gender_deploy.prototxt")
    
    return model_best, face_cascade, ageNet, genderNet
    
model_best, face_cascade, ageNet, genderNet = load_models()

class_names = ['Angry', 'Disgusted', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60+)']
genderList = ['Male', 'Female']
modelMeanValues = (78.4263377603, 87.7689143744, 114.895847746)

output_file = "person_details.csv"

# Initialize CSV file with headers
if not os.path.exists(output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Gender", "Age", "Emotion"])

def run_detection():
    cap = cv2.VideoCapture(0)
    st_frame = st.empty()  # Placeholder for video frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to capture video frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            if face.size == 0:
                continue

            # Prepare for age & gender detection
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), modelMeanValues, swapRB=False)
            genderNet.setInput(blob)
            gender = genderList[genderNet.forward()[0].argmax()]
            ageNet.setInput(blob)
            age = ageList[ageNet.forward()[0].argmax()]

            # Emotion detection preparation
            face_resized = cv2.resize(face, (48, 48))
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            face_array = img_to_array(face_gray) / 255.0
            face_array = np.expand_dims(face_array, axis=0)
            emotion = class_names[np.argmax(model_best.predict(face_array))]

            # Add text to frame
            label = f"{gender}, {age}, {emotion}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Save details to CSV
            with open(output_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([gender, age, emotion])

        # Display frame in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st_frame.image(frame_rgb, channels="RGB")

        if st.session_state.stop_camera:
            cap.release()
            break

# Analyze CSV function
def analyze_csv():
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        st.sidebar.subheader("CSV Analysis")
        st.sidebar.write(df)
        
        # Show summary statistics
        st.sidebar.write("### Summary Statistics")
        st.sidebar.write(df.describe(include='all'))
        
        # Visualize distributions
        st.sidebar.write("### Gender Distribution")
        st.sidebar.bar_chart(df['Gender'].value_counts())
        
        st.sidebar.write("### Age Distribution")
        st.sidebar.line_chart(df['Age'].value_counts())
        
        st.sidebar.write("### Emotion Distribution")
        st.sidebar.bar_chart(df['Emotion'].value_counts())
    else:
        st.sidebar.warning("No CSV file found. Please start detection first.")

st.title("👥 Feedback system")
st.write("Click the button to start/stop the camera and record results to a CSV file.")

if 'stop_camera' not in st.session_state:
    st.session_state.stop_camera = True

if st.session_state.stop_camera:
    if st.button("Start Camera"):
        st.session_state.stop_camera = False
        run_detection()
else:
    if st.button("Stop Camera"):
        st.session_state.stop_camera = True
        st.success("Camera stopped.")

# Add 'Analyze CSV' button to the sidebar
if st.sidebar.button("Analyze CSV"):
    analyze_csv()
