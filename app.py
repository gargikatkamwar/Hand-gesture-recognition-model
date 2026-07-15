import streamlit as st
import joblib
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image

MODEL_PATH = 'gesture_classifier.pkl'

st.set_page_config(page_title="Hand Gesture Recognition", page_icon="✋")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


mp_hands = mp.solutions.hands


def extract_landmarks(img_bgr, hands):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if not result.multi_hand_landmarks:
        return None

    lm = result.multi_hand_landmarks[0].landmark
    coords = np.array([[p.x, p.y, p.z] for p in lm])
    coords -= coords[0]
    scale = np.linalg.norm(coords[9])
    coords /= (scale if scale > 0 else 1)
    return coords.flatten()


st.title("✋ Hand Gesture Recognition")
st.write("Take a photo or upload an image to identify the hand gesture.")

clf = load_model()

tab1, tab2 = st.tabs(["📷 Use Camera", "📁 Upload Image"])

image_to_process = None

with tab1:
    camera_photo = st.camera_input("Take a photo of your hand gesture")
    if camera_photo is not None:
        image_to_process = Image.open(camera_photo)

with tab2:
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_to_process = Image.open(uploaded_file)

if image_to_process is not None:
    img_array = np.array(image_to_process.convert("RGB"))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                         min_detection_confidence=0.5) as hands:
        feats = extract_landmarks(img_bgr, hands)

    if feats is not None:
        prediction = clf.predict([feats])[0]
        st.success(f"**Predicted gesture: {prediction}**")
    else:
        st.warning("No hand detected. Try again with your hand clearly visible and well lit.")
else:
    st.info("Use the camera tab or upload an image to get a prediction.")
