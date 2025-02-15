import cv2
import os
import numpy as np

def extract_keypoints(input_folder, output_folder):
    """Detects and stores keypoints and descriptors from images."""
    
    
    input_folder = "data/raw/Cassini_LEO_Corkscrew_1.00"
    output_folder = "data/processed/keypoints"
    
    os.makedirs(output_folder, exist_ok=True)

    # SIFT Detector
    sift = cv2.SIFT_create()

    # List images
    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".png")])

    for i, image_file in enumerate(image_files):
        img_path = os.path.join(input_folder, image_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Error: Unable to load {image_file}")
            continue

        # Detect keypoints & descriptors
        keypoints, descriptors = sift.detectAndCompute(img, None)

        # Save keypoints & descriptors
        keypoints_file = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}.npz")
        np.savez_compressed(keypoints_file, 
                            keypoints=[(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints],
                            descriptors=descriptors)

        print(f"Extracted features {i+1}/{len(image_files)}: {image_file}")

    print("Feature detection complete.")

