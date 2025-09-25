import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

st.title("😷 Face Mask Detection (YOLOv5)")
st.markdown("Upload an **Image**, **Video**, or use **Webcam** to detect `With Mask` / `Without Mask`.")

# ---------------- Load YOLOv5 model ----------------
@st.cache_resource
def load_model():
    model = torch.hub.load(
        'ultralytics/yolov5',
        'custom',
        path='mask_yolov5.pt',   # your trained weights
        source='github'
    )
    model.conf = 0.5  # confidence threshold
    return model

model = load_model()

# ---------------- Mode Selection ----------------
mode = st.radio(
    "Select Input Mode",
    ["Image", "Video", "Webcam"]
)

# ---------------- IMAGE MODE ----------------
if mode == "Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        img = np.array(img)

        # Run inference
        results = model(img)
        annotated = np.squeeze(results.render())  # draw boxes
        st.image(annotated, channels="RGB", caption="Detection Result")

        # 🔢 Detection Summary
        names = model.names
        labels = results.xyxy[0][:, -1].cpu().numpy().astype(int)
        counts = {names[int(c)]: (labels == c).sum() for c in np.unique(labels)}
        st.subheader("🔢 Detection Summary")
        st.write(counts if counts else "No faces detected.")

# ---------------- VIDEO MODE ----------------
elif mode == "Video":
    uploaded_file = st.file_uploader("Upload a Video", type=["mp4"])
    if uploaded_file:
        # Save uploaded video to a temp file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()
        summary_placeholder = st.empty()  # placeholder for summary
        total_counts = {}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            annotated_frame = np.squeeze(results.render())
            stframe.image(annotated_frame, channels="BGR")

            # Update summary counts
            names = model.names
            labels = results.xyxy[0][:, -1].cpu().numpy().astype(int)
            for c in np.unique(labels):
                name = names[int(c)]
                total_counts[name] = total_counts.get(name, 0) + (labels == c).sum()
            summary_placeholder.write(f"🔢 **Detection Summary:** {total_counts}")

        cap.release()
        os.unlink(video_path)

# ---------------- WEBCAM MODE ----------------
elif mode == "Webcam":
    st.warning("📸 Click **Start** to begin webcam detection.")
    run_webcam = st.checkbox("Start Webcam")
    if run_webcam:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        summary_placeholder = st.empty()
        total_counts = {}

        while run_webcam:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            annotated_frame = np.squeeze(results.render())
            stframe.image(annotated_frame, channels="BGR")

            # Update summary counts
            names = model.names
            labels = results.xyxy[0][:, -1].cpu().numpy().astype(int)
            for c in np.unique(labels):
                name = names[int(c)]
                total_counts[name] = total_counts.get(name, 0) + (labels == c).sum()
            summary_placeholder.write(f"🔢 **Detection Summary:** {total_counts}")

        cap.release()
