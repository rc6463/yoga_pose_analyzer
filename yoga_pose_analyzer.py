#﻿import cv2
#import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image as PILImage
import tempfile

st.title(" Cowboy Zen Yoga Pose Analyzer")

uploaded_file = st.file_uploader("Upload a Yoga Pose Image", type=['jpg', 'jpeg', 'png'])

# Helper functions
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return round(np.degrees(angle), 2)

def check_angle(angle, joint_name):
    flags = {
        'R_Knee_Angle': (160, 180),
        'L_Knee_Angle': (160, 180),
        'R_Elbow_Angle': (80, 100),
        'L_Elbow_Angle': (80, 100),
        'Hip_Spine_Angle': (150, 180),
        'Neck_Angle': (120, 140),
        'R_Ankle_Angle': (90, 100),
        'L_Ankle_Angle': (90, 100)
    }
    low, high = flags[joint_name]
    return "❌ Bad" if angle < low or angle > high else "✅ Good"

# MediaPipe setup
#mp_pose = mp.solutions.pose
#pose = mp_pose.Pose(static_image_mode=True)
#mp_drawing = mp.solutions.drawing_utils

# Process uploaded file
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    image = cv2.imread(tmp_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        h, w = image.shape[:2]
        landmarks = results.pose_landmarks.landmark

        def get_point(idx):
            lm = landmarks[idx]
            return [lm.x * w, lm.y * h]

        points = {
            'right_hip': get_point(mp_pose.PoseLandmark.RIGHT_HIP.value),
            'right_knee': get_point(mp_pose.PoseLandmark.RIGHT_KNEE.value),
            'right_ankle': get_point(mp_pose.PoseLandmark.RIGHT_ANKLE.value),
            'right_foot': get_point(mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value),
            'left_hip': get_point(mp_pose.PoseLandmark.LEFT_HIP.value),
            'left_knee': get_point(mp_pose.PoseLandmark.LEFT_KNEE.value),
            'left_ankle': get_point(mp_pose.PoseLandmark.LEFT_ANKLE.value),
            'left_foot': get_point(mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value),
            'right_shoulder': get_point(mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
            'right_elbow': get_point(mp_pose.PoseLandmark.RIGHT_ELBOW.value),
            'right_wrist': get_point(mp_pose.PoseLandmark.RIGHT_WRIST.value),
            'left_shoulder': get_point(mp_pose.PoseLandmark.LEFT_SHOULDER.value),
            'left_elbow': get_point(mp_pose.PoseLandmark.LEFT_ELBOW.value),
            'left_wrist': get_point(mp_pose.PoseLandmark.LEFT_WRIST.value),
            'left_ear': get_point(mp_pose.PoseLandmark.LEFT_EAR.value),
            'right_ear': get_point(mp_pose.PoseLandmark.RIGHT_EAR.value),
        }

        # Calculate angles
        angles = {}
        angles['R_Knee_Angle'] = calculate_angle(points['right_hip'], points['right_knee'], points['right_ankle'])
        angles['L_Knee_Angle'] = calculate_angle(points['left_hip'], points['left_knee'], points['left_ankle'])
        angles['R_Elbow_Angle'] = calculate_angle(points['right_shoulder'], points['right_elbow'], points['right_wrist'])
        angles['L_Elbow_Angle'] = calculate_angle(points['left_shoulder'], points['left_elbow'], points['left_wrist'])
        angles['Hip_Spine_Angle'] = calculate_angle(points['right_shoulder'], points['right_hip'], points['right_knee'])
        angles['Neck_Angle'] = calculate_angle(points['left_shoulder'], points['left_ear'], points['right_shoulder'])
        angles['R_Ankle_Angle'] = calculate_angle(points['right_knee'], points['right_ankle'], points['right_foot'])
        angles['L_Ankle_Angle'] = calculate_angle(points['left_knee'], points['left_ankle'], points['left_foot'])

        # Display feedback
        st.subheader("Pose Feedback")
        for k, v in angles.items():
            st.write(f"{k}: {v} ({check_angle(v, k)})")

        # Draw landmarks
        #mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
       # image_pil = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
       # st.image(image_pil, caption="Pose with Keypoints", use_column_width=True)
    else:
        st.error("No pose detected. Try another image.")

