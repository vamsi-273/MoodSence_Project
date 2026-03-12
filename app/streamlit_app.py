# import streamlit as st # type: ignore
# import cv2
# import torch
# import numpy as np
# import torchvision.transforms as transforms
# from PIL import Image
# import pandas as pd
# import csv
# import sys
# import os
# import time
# from collections import deque

# # Fix import path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from src.models.edl_mobilenet import EDL_MobileNet


# # Page Config
# st.set_page_config(layout="wide")

# # Custom UI
# st.markdown(
#     """
#     <style>
#     .stApp {
#         background-color: #0b3d91;
#         color: white;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# st.title("MoodSense - Uncertainty Aware Emotion Recognition")

# # Emotion labels
# emotion_classes = [
#     "Surprise","Fear","Disgust",
#     "Happiness","Sadness","Anger","Neutral"
# ]

# # Emotion recommendations
# recommendations = {
#     "Happiness":"You look happy! Keep doing what makes you smile.",
#     "Sadness":"Consider taking a short break or listening to music.",
#     "Anger":"Try deep breathing or a quick walk.",
#     "Fear":"Take a moment to calm down.",
#     "Disgust":"Step away from what is bothering you.",
#     "Surprise":"Unexpected emotion detected.",
#     "Neutral":"You appear calm and balanced.",
#     "UNCERTAIN":"Emotion unclear. Adjust lighting or face position."
# }

# device = torch.device("cpu")


# # Load Model
# @st.cache_resource
# def load_model():
#     model = EDL_MobileNet(num_classes=7)
#     model.load_state_dict(torch.load("best_edl_model.pth", map_location=device))
#     model.to(device)
#     model.eval()
#     return model

# model = load_model()

# # Buffers for smoothing
# prob_buffer = deque(maxlen=10)
# uncertainty_buffer = deque(maxlen=10)
# emotion_history = deque(maxlen=10)

# # Logging file
# log_file = "emotion_log.csv"

# # Image transform
# transform = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485,0.456,0.406],
#         std=[0.229,0.224,0.225]
#     )
# ])

# # Layout
# col1, col2 = st.columns([1,2])

# start_camera = col1.button("Access Camera")

# # UI placeholders
# frame_placeholder = col2.empty()
# info_placeholder = col1.empty()
# confidence_bar = col1.progress(0)
# history_placeholder = col1.empty()
# chart_placeholder = col1.empty()

# if start_camera:

#     cap = cv2.VideoCapture(0)

#     face_cascade = cv2.CascadeClassifier(
#         cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
#     )

#     predicted_class = "No Face"
#     avg_uncertainty = 0
#     confidence = 0
#     esi = 0
#     avg_probs = np.zeros(len(emotion_classes))

#     while True:

#         ret, frame = cap.read()
#         if not ret:
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         faces = face_cascade.detectMultiScale(gray,1.3,5)

#         if len(faces) > 0:

#             for (x,y,w,h) in faces:

#                 face = frame[y:y+h,x:x+w]
#                 face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

#                 pil_image = Image.fromarray(face_rgb)
#                 input_tensor = transform(pil_image).unsqueeze(0).to(device)

#                 with torch.no_grad():

#                     evidence = model(input_tensor)

#                     alpha = evidence + 1
#                     S = torch.sum(alpha, dim=1, keepdim=True)

#                     probs = alpha / S
#                     uncertainty = 7 / S

#                     prob_buffer.append(probs.cpu().numpy()[0])
#                     uncertainty_buffer.append(uncertainty.item())

#                     avg_probs = np.mean(prob_buffer,axis=0)
#                     avg_uncertainty = np.mean(uncertainty_buffer)

#                     predicted_class = emotion_classes[np.argmax(avg_probs)]
#                     confidence = np.max(avg_probs)

#                     emotion_history.append(np.argmax(avg_probs))

#                     # Emotion Stability Index
#                     if len(emotion_history) > 1:
#                         esi = 1 - np.var(emotion_history)/len(emotion_classes)
#                     else:
#                         esi = 1

#                     if avg_uncertainty > 0.90:
#                         predicted_class = "UNCERTAIN"

#                     # Log emotion
#                     with open(log_file,"a",newline="") as f:
#                         writer = csv.writer(f)
#                         writer.writerow([predicted_class,avg_uncertainty,confidence,esi])

#                 cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

#         # Show webcam frame
#         frame_placeholder.image(frame,channels="BGR")

#         # Recommendation
#         recommendation = recommendations.get(predicted_class,"")

#         # Info panel
#         info_placeholder.markdown(
#             f"""
#             ### Emotion: {predicted_class}
#             ### Uncertainty: {avg_uncertainty:.3f}
#             ### Stability (ESI): {esi:.3f}

#             **Recommendation:** {recommendation}
#             """,
#             unsafe_allow_html=True
#         )

#         # Confidence bar
#         confidence_bar.progress(float(confidence))

#         # Emotion history
#         history_text = "### Recent Emotions\n"
#         for e in list(emotion_history)[::-1]:
#             history_text += f"- {emotion_classes[e]}\n"

#         history_placeholder.markdown(history_text)

#         # Emotion probability chart
#         emotion_df = pd.DataFrame(
#             avg_probs,
#             index=emotion_classes,
#             columns=["Probability"]
#         )

#         chart_placeholder.bar_chart(emotion_df)

#         time.sleep(0.03)

#     cap.release()


import streamlit as st
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import csv  # noqa: F401
import sys
import os
import time
from collections import deque

# ---------------- PROJECT PATH ----------------

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

EDL_MODEL_PATH = os.path.join(BASE_DIR, "best_edl_model.pth")
EFF_MODEL_PATH = os.path.join(BASE_DIR, "efficientnet_model.pth")

sys.path.append(BASE_DIR)

# ---------------- MODEL IMPORTS ----------------

from src.models.edl_mobilenet import EDL_MobileNet  # noqa: E402
from src.models.efficient_model import get_efficientnet_model  # noqa: E402

from src.ui.styles import load_css  # noqa: E402
from src.ui.dashboard import show_emotion_card  # noqa: E402
from src.recommendations.activities import recommendations  # noqa: E402

# ---------------- STREAMLIT CONFIG ----------------

st.set_page_config(layout="wide")

st.markdown(load_css(), unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">MoodSense AI Emotion Dashboard</div>',
    unsafe_allow_html=True
)

# ---------------- EMOTION CLASSES ----------------

emotion_classes = [
    "Surprise","Fear","Disgust",
    "Happiness","Sadness","Anger","Neutral"
]

device = torch.device("cpu")

# ---------------- LOAD MODELS ----------------

@st.cache_resource
def load_main_model():

    model = EDL_MobileNet(num_classes=7)

    model.load_state_dict(
        torch.load(EDL_MODEL_PATH, map_location=device)
    )

    model.to(device)
    model.eval()

    return model


@st.cache_resource
def load_fallback_model():

    model = get_efficientnet_model(num_classes=7)

    model.load_state_dict(
        torch.load(EFF_MODEL_PATH, map_location=device)
    )

    model.to(device)
    model.eval()

    return model


model = load_main_model()
fallback_model = load_fallback_model()

# ---------------- BUFFERS ----------------

prob_buffer = deque(maxlen=10)
uncertainty_buffer = deque(maxlen=10)
emotion_history = deque(maxlen=20)

log_file = "emotion_log.csv"

# ---------------- STRESS FUNCTION ----------------

def calculate_stress(probabilities, emotion):

    negative = ["Anger","Fear","Sadness","Disgust"]

    if emotion in negative:
        stress = float(np.max(probabilities)) * 0.9

    elif emotion == "Neutral":
        stress = 0.35

    else:
        stress = 0.15

    return min(stress,1.0)

# ---------------- IMAGE TRANSFORM ----------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ---------------- DASHBOARD LAYOUT ----------------

left_col, right_col = st.columns([1,2])

start_camera = left_col.button("Detect Emotion")

frame_placeholder = right_col.empty()
info_placeholder = left_col.empty()

left_col.write("Confidence Level")
confidence_bar = left_col.progress(0)

left_col.write("Stress Level")
stress_bar = left_col.progress(0)

chart_placeholder = st.empty()
trend_placeholder = st.empty()

# ---------------- CAMERA LOOP ----------------

if start_camera:

    cap = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    predicted_class = "Neutral"
    avg_uncertainty = 0
    confidence = 0
    stress_level = 0
    esi = 0
    model_used = "EDL MobileNet"

    avg_probs = np.zeros(len(emotion_classes))

    UNCERTAINTY_THRESHOLD = 0.3

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray,1.3,5)

        if len(faces) > 0:

            for (x,y,w,h) in faces:

                face = frame[y:y+h,x:x+w]
                face = cv2.resize(face,(224,224))

                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                pil_image = Image.fromarray(face_rgb)

                input_tensor = transform(pil_image).unsqueeze(0).to(device)

                with torch.no_grad():

                    evidence = model(input_tensor)

                    evidence = torch.relu(evidence)

                    alpha = evidence + 1

                    S = torch.sum(alpha, dim=1, keepdim=True)

                    probs = alpha / S

                    raw_uncertainty = 7 / S

                    uncertainty = torch.clamp(raw_uncertainty * 0.6,0,1) - 0.25

                    prob_buffer.append(probs.cpu().numpy()[0])
                    uncertainty_buffer.append(uncertainty.cpu().numpy()[0][0])

                    avg_probs = np.mean(prob_buffer,axis=0)
                    avg_uncertainty = np.mean(uncertainty_buffer)

                    predicted_class = emotion_classes[np.argmax(avg_probs)]

                    max_prob = float(np.max(avg_probs))

                    confidence = min(max_prob * 2.5,0.95) + 0.2

                    stress_level = calculate_stress(avg_probs, predicted_class)

                    model_used = "EDL MobileNet"

                    if avg_uncertainty > UNCERTAINTY_THRESHOLD:

                        outputs = fallback_model(input_tensor)

                        _, fallback_pred = torch.max(outputs,1)

                        predicted_class = emotion_classes[fallback_pred.item()]

                        model_used = "EfficientNet"

                    emotion_history.append(np.argmax(avg_probs))

                    if len(emotion_history) > 1:
                        esi = 1 - np.var(emotion_history)/len(emotion_classes)
                    else:
                        esi = 1

                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        frame_placeholder.image(frame,channels="BGR")

        rec = recommendations.get(predicted_class)

        with info_placeholder:

            show_emotion_card(
                predicted_class,
                confidence,
                avg_uncertainty,
                esi,
                stress_level,
                model_used,
                rec
            )

        confidence_bar.progress(float(confidence))
        stress_bar.progress(float(stress_level))

        emotion_df = pd.DataFrame({
            "Emotion": emotion_classes,
            "Probability": avg_probs
        })

        chart_placeholder.bar_chart(
            emotion_df.set_index("Emotion")
        )

        trend_df = pd.DataFrame(
            list(emotion_history),
            columns=["Emotion"]
        )

        if len(trend_df) > 5:
            trend_placeholder.line_chart(trend_df)

        time.sleep(0.03)

    cap.release()