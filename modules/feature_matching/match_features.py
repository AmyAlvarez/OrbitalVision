import cv2
import numpy as np
import os
import json  # To save matched points and paths

# Paths
keypoints_folder = "./keypoints_data"
processed_folder = "./processed_images"
matches_folder = "./feature_matches"
matches_data_file = "./feature_matches/matches.json"  # To save matched points and paths

# Create output folder for matches
os.makedirs(matches_folder, exist_ok=True)

# List all keypoints files
keypoints_files = sorted([f for f in os.listdir(keypoints_folder) if f.endswith(".npz")])

# Debug: Inspect first keypoints file
first_file = os.path.join(keypoints_folder, keypoints_files[0])
data = np.load(first_file)
print(f"Inspecting keypoints from {keypoints_files[0]}:")
print(f"First keypoint: {data['keypoints'][0]}")
print(f"Number of attributes in each keypoint: {len(data['keypoints'][0])}")

# Initialize Brute-Force Matcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# To store matched points and image paths
matches_data = []

# Loop through consecutive frames
for i in range(len(keypoints_files) - 1):
    # Load keypoints and descriptors for consecutive frames
    data1 = np.load(os.path.join(keypoints_folder, keypoints_files[i]))
    data2 = np.load(os.path.join(keypoints_folder, keypoints_files[i + 1]))

    keypoints_data1 = data1["keypoints"]
    descriptors1 = data1["descriptors"]
    keypoints_data2 = data2["keypoints"]
    descriptors2 = data2["descriptors"]

    # Debug: Print keypoint data structure
    print(f"First keypoint data: {keypoints_data1[0]}")
    
    # Reconstruct keypoints from saved data
    keypoints1 = []
    for kp in keypoints_data1:
        try:
            keypoint = cv2.KeyPoint(
                x=float(kp[0]),
                y=float(kp[1]),
                size=float(kp[2]),  # This is the required parameter
                angle=float(kp[3]),
                response=float(kp[4]),
                octave=int(kp[5]),
                class_id=int(kp[6])
            )
            keypoints1.append(keypoint)
        except Exception as e:
            print(f"Error creating keypoint: {e}")
            print(f"Keypoint data: {kp}")
            raise

    keypoints2 = []
    for kp in keypoints_data2:
        keypoint = cv2.KeyPoint(
            x=float(kp[0]),
            y=float(kp[1]),
            size=float(kp[2]),
            angle=float(kp[3]),
            response=float(kp[4]),
            octave=int(kp[5]),
            class_id=int(kp[6])
        )
        keypoints2.append(keypoint)

    # Perform feature matching
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched points
    points1 = np.array([keypoints1[m.queryIdx].pt for m in matches], dtype=np.float32)
    points2 = np.array([keypoints2[m.trainIdx].pt for m in matches], dtype=np.float32)

    # Save matched points and image paths
    matches_data.append({
        "image1": os.path.join(processed_folder, f"processed_{keypoints_files[i].replace('.npz', '.png')}"),
        "image2": os.path.join(processed_folder, f"processed_{keypoints_files[i + 1].replace('.npz', '.png')}"),
        "points1": points1.tolist(),
        "points2": points2.tolist()
    })

    # Visualize the top matches
    img1 = cv2.imread(os.path.join(processed_folder, f"processed_{keypoints_files[i].replace('.npz', '.png')}"))
    img2 = cv2.imread(os.path.join(processed_folder, f"processed_{keypoints_files[i + 1].replace('.npz', '.png')}"))

    if img1 is None or img2 is None:
        print(f"Error loading processed images for visualization: {keypoints_files[i]} or {keypoints_files[i + 1]}")
        continue

    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Save the visualization
    output_path = os.path.join(matches_folder, f"feature_matches_{i + 1}.png")
    cv2.imwrite(output_path, img_matches)
    print(f"Saved feature match visualization: {output_path}")

# Save matched points and paths to a JSON file
with open(matches_data_file, "w") as f:
    json.dump(matches_data, f, indent=4)

print(f"Feature matching data saved to {matches_data_file}.")
print("Feature matching complete!")
