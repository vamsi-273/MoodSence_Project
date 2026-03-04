import streamlit as st # type: ignore
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import csv
import sys
import os
import time
from collections import deque

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.edl_mobilenet import EDL_MobileNet


# Page Config
st.set_page_config(layout="wide")

# Custom UI
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b3d91;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("MoodSense - Uncertainty Aware Emotion Recognition")

# Emotion labels
emotion_classes = [
    "Surprise","Fear","Disgust",
    "Happiness","Sadness","Anger","Neutral"
]

# Emotion recommendations
recommendations = {
    "Happiness":"You look happy! Keep doing what makes you smile.",
    "Sadness":"Consider taking a short break or listening to music.",
    "Anger":"Try deep breathing or a quick walk.",
    "Fear":"Take a moment to calm down.",
    "Disgust":"Step away from what is bothering you.",
    "Surprise":"Unexpected emotion detected.",
    "Neutral":"You appear calm and balanced.",
    "UNCERTAIN":"Emotion unclear. Adjust lighting or face position."
}

device = torch.device("cpu")


# Load Model
@st.cache_resource
def load_model():
    model = EDL_MobileNet(num_classes=7)
    model.load_state_dict(torch.load("best_edl_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# Buffers for smoothing
prob_buffer = deque(maxlen=10)
uncertainty_buffer = deque(maxlen=10)
emotion_history = deque(maxlen=10)

# Logging file
log_file = "emotion_log.csv"

# Image transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# Layout
col1, col2 = st.columns([1,2])

start_camera = col1.button("Access Camera")

# UI placeholders
frame_placeholder = col2.empty()
info_placeholder = col1.empty()
confidence_bar = col1.progress(0)
history_placeholder = col1.empty()
chart_placeholder = col1.empty()

if start_camera:

    cap = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    predicted_class = "No Face"
    avg_uncertainty = 0
    confidence = 0
    esi = 0
    avg_probs = np.zeros(len(emotion_classes))

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray,1.3,5)

        if len(faces) > 0:

            for (x,y,w,h) in faces:

                face = frame[y:y+h,x:x+w]
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                pil_image = Image.fromarray(face_rgb)
                input_tensor = transform(pil_image).unsqueeze(0).to(device)

                with torch.no_grad():

                    evidence = model(input_tensor)

                    alpha = evidence + 1
                    S = torch.sum(alpha, dim=1, keepdim=True)

                    probs = alpha / S
                    uncertainty = 7 / S

                    prob_buffer.append(probs.cpu().numpy()[0])
                    uncertainty_buffer.append(uncertainty.item())

                    avg_probs = np.mean(prob_buffer,axis=0)
                    avg_uncertainty = np.mean(uncertainty_buffer)

                    predicted_class = emotion_classes[np.argmax(avg_probs)]
                    confidence = np.max(avg_probs)

                    emotion_history.append(np.argmax(avg_probs))

                    # Emotion Stability Index
                    if len(emotion_history) > 1:
                        esi = 1 - np.var(emotion_history)/len(emotion_classes)
                    else:
                        esi = 1

                    if avg_uncertainty > 0.90:
                        predicted_class = "UNCERTAIN"

                    # Log emotion
                    with open(log_file,"a",newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([predicted_class,avg_uncertainty,confidence,esi])

                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        # Show webcam frame
        frame_placeholder.image(frame,channels="BGR")

        # Recommendation
        recommendation = recommendations.get(predicted_class,"")

        # Info panel
        info_placeholder.markdown(
            f"""
            ### Emotion: {predicted_class}
            ### Uncertainty: {avg_uncertainty:.3f}
            ### Stability (ESI): {esi:.3f}

            **Recommendation:** {recommendation}
            """,
            unsafe_allow_html=True
        )

        # Confidence bar
        confidence_bar.progress(float(confidence))

        # Emotion history
        history_text = "### Recent Emotions\n"
        for e in list(emotion_history)[::-1]:
            history_text += f"- {emotion_classes[e]}\n"

        history_placeholder.markdown(history_text)

        # Emotion probability chart
        emotion_df = pd.DataFrame(
            avg_probs,
            index=emotion_classes,
            columns=["Probability"]
        )

        chart_placeholder.bar_chart(emotion_df)

        time.sleep(0.03)

    cap.release()