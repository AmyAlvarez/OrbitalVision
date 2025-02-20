import os
import cv2
import numpy as np

# Import modules
from modules.preprocessing.preprocess import preprocess_images
from modules.feature_detection.detect import extract_keypoints
from modules.feature_matching.match_features import match_features

# Define paths
RAW_DATA_FOLDER = "data/raw/frames"
PROCESSED_DATA_FOLDER = "data/processed/images"
KEYPOINTS_FOLDER = "data/processed/keypoints"
MATCHES_FOLDER = "data/processed/matches"

# Create necessary directories
os.makedirs(PROCESSED_DATA_FOLDER, exist_ok=True)
os.makedirs(KEYPOINTS_FOLDER, exist_ok=True)
os.makedirs(MATCHES_FOLDER, exist_ok=True)

## Preprocess Images (Contrast, CLAHE, Edge Detection)
print("\n Running Image Preprocessing...")
preprocess_images(RAW_DATA_FOLDER, PROCESSED_DATA_FOLDER)

### Extract Keypoints & Descriptors
print("\n Running Feature Detection...")
extract_keypoints(PROCESSED_DATA_FOLDER, KEYPOINTS_FOLDER)

### Match Features Between Frames
print("\n Running Feature Matching...")
match_features(KEYPOINTS_FOLDER, PROCESSED_DATA_FOLDER, MATCHES_FOLDER)

print("\n Complete. All processed data is saved in the structured directories.")
