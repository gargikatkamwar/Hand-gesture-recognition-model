import joblib
import mediapipe as mp
import cv2
import numpy as np
import argparse
import os

# Load the model
MODEL_PATH = 'gesture_classifier (2).pkl'
try:
    loaded_clf = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}!")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    exit(1)

mp_hands = mp.solutions.hands

def extract_landmarks(image_path, hands):
    """
    Extracts hand landmarks from an image file.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return None
    
    # Convert the image to RGB for MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    result = hands.process(img_rgb)
    
    if not result.multi_hand_landmarks:
        print("No hand detected in the image.")
        return None
        
    lm = result.multi_hand_landmarks[0].landmark
    coords = np.array([[p.x, p.y, p.z] for p in lm])
    
    # Normalize landmarks
    coords -= coords[0] # Translate to origin
    scale = np.linalg.norm(coords[9])
    coords /= (scale if scale > 0 else 1) # Scale to make it robust to distance
    
    return coords.flatten()

def main():
    parser = argparse.ArgumentParser(description='Predict gesture from an image.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image file.')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found at {args.image_path}")
        return

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        print(f"Processing image: {args.image_path}")
        feats = extract_landmarks(args.image_path, hands)
        
        if feats is not None:
            # Make prediction
            prediction = loaded_clf.predict([feats])[0]
            print(f"Predicted gesture: {prediction}")
        else:
            print("Could not make a prediction.")

if __name__ == '__main__':
    main()
