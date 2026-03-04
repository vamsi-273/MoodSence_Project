import cv2
import torch
# import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from src.models.edl_mobilenet import EDL_MobileNet


# Emotion labels
emotion_classes = [
    "Surprise",
    "Fear",
    "Disgust",
    "Happiness",
    "Sadness",
    "Anger",
    "Neutral"
]


# Load model
device = torch.device("cpu")
model = EDL_MobileNet(num_classes=7)
model.load_state_dict(torch.load("best_edl_model.pth", map_location=device))
model.to(device)
model.eval()


# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Load face detector
face_cascade = cv2.CascadeClassifier(
    "app/haarcascade_frontalface_default.xml"
)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(face_rgb)
        input_tensor = transform(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            evidence = model(input_tensor)
            alpha = evidence + 1
            S = torch.sum(alpha, dim=1, keepdim=True)

            probs = alpha / S
            uncertainty = 7 / S

            confidence, predicted = torch.max(probs, 1)

            predicted_class = emotion_classes[predicted.item()]
            uncertainty_value = uncertainty.item()

        # Decision threshold
        if uncertainty_value > 0.87:
            label = "UNCERTAIN"
            color = (0, 0, 255)
        else:
            label = f"{predicted_class} ({confidence.item():.2f})"
            color = (0, 255, 0)

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Put text
        cv2.putText(
            frame,
            f"{label} | U:{uncertainty_value:.2f}",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    cv2.imshow("MoodSense - Uncertainty Aware", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()