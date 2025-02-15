import json
import cv2
import numpy as np

# Define camera intrinsic parameters
# Replace these values with your camera's actual parameters
fx = 1000  # focal length in x direction (in pixels)
fy = 1000  # focal length in y direction (in pixels)
cx = 960   # principal point x coordinate (usually image_width/2)
cy = 540   # principal point y coordinate (usually image_height/2)

# Create intrinsic camera matrix
intrinsic_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])

# Load matches data
with open("./feature_matches/matches.json", "r") as f:
    matches_data = json.load(f)

# Loop through matched data
for match in matches_data:
    image1_path = match["image1"]
    image2_path = match["image2"]
    points1 = np.array(match["points1"], dtype=np.float32)
    points2 = np.array(match["points2"], dtype=np.float32)

    # Compute Essential Matrix
    essential_matrix, mask = cv2.findEssentialMat(
        points1, points2, intrinsic_matrix, 
        method=cv2.RANSAC, prob=0.999, threshold=1.0
    )

    # Recover pose
    _, R, T, mask_pose = cv2.recoverPose(
        essential_matrix, points1, points2, intrinsic_matrix
    )

    print(f"Pose between {image1_path} and {image2_path}:")
    print("Rotation:\n", R)
    print("Translation:\n", T)

def load_matches_data(matches_file):
    """Load the matched points from the JSON file"""
    with open(matches_file, 'r') as f:
        return json.load(f)

def estimate_pose(points1, points2, K):
    """
    Estimate the relative pose between two images
    Args:
        points1, points2: Matched points in both images (Nx2 arrays)
        K: Camera intrinsic matrix (3x3 array)
    Returns:
        R: Rotation matrix
        t: Translation vector
        mask: Inlier mask
    """
    # Convert points to numpy arrays if they aren't already
    points1 = np.array(points1)
    points2 = np.array(points2)
    K = np.array(K)

    # Compute Essential Matrix
    essential_matrix, mask = cv2.findEssentialMat(points1, points2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    # Recover pose
    _, R, t, mask_pose = cv2.recoverPose(essential_matrix, points1, points2, K)

    return R, t, mask_pose

import cv2
import numpy as np
import os
from pathlib import Path

def match_features(img1_path, img2_path, keypoints1, keypoints2, descriptors1, descriptors2):
    # Create BFMatcher and match descriptors
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Filter good matches using distance threshold
    good_matches = []
    min_dist = matches[0].distance
    for match in matches:
        if match.distance < 2 * min_dist:
            good_matches.append(match)
    
    # Extract matched points
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])
    
    return points1, points2, good_matches

def process_image_sequence(image_folder, keypoints_folder):
    # Get sorted lists of images and keypoint files
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))])
    keypoints_files = sorted([f for f in os.listdir(keypoints_folder) if f.endswith('.npz')])
    
    # Store all matched points and image pairs
    all_matches = []
    
    for i in range(len(image_files) - 1):
        # Load consecutive images
        img1_path = os.path.join(image_folder, image_files[i])
        img2_path = os.path.join(image_folder, image_files[i + 1])
        
        # Load keypoints and descriptors
        kp_data1 = np.load(os.path.join(keypoints_folder, keypoints_files[i]))
        kp_data2 = np.load(os.path.join(keypoints_folder, keypoints_files[i + 1]))
        
        # Reconstruct keypoints
        keypoints1 = [
            cv2.KeyPoint(
                x=float(kp[0]), y=float(kp[1]), 
                size=float(kp[2]), angle=float(kp[3]),
                response=float(kp[4]), octave=int(kp[5]),
                class_id=int(kp[6])
            ) for kp in kp_data1['keypoints']
        ]
        keypoints2 = [
            cv2.KeyPoint(
                x=float(kp[0]), y=float(kp[1]), 
                size=float(kp[2]), angle=float(kp[3]),
                response=float(kp[4]), octave=int(kp[5]),
                class_id=int(kp[6])
            ) for kp in kp_data2['keypoints']
        ]
        
        # Get descriptors
        descriptors1 = kp_data1['descriptors']
        descriptors2 = kp_data2['descriptors']
        
        # Match features and get matched points
        points1, points2, matches = match_features(
            img1_path, img2_path,
            keypoints1, keypoints2,
            descriptors1, descriptors2
        )
        
        # Store the matches information
        match_info = {
            'image1_path': img1_path,
            'image2_path': img2_path,
            'points1': points1,
            'points2': points2,
            'num_matches': len(matches)
        }
        all_matches.append(match_info)
        
        # Optional: Visualize matches
        if True:  # Set to False to disable visualization
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            img_matches = cv2.drawMatches(
                img1, keypoints1, img2, keypoints2, 
                matches[:50], None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            cv2.imshow(f'Matches {i}-{i+1}', img_matches)
            cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    return all_matches

if __name__ == "__main__":
    # Update paths to match your new folder names
    image_folder = "pose_est_images"  # Changed from "images"
    keypoints_folder = "keypoints_images"  # Changed from "keypoints"
    
    # Process all images and get matches
    matches_data = process_image_sequence(image_folder, keypoints_folder)
    
    # Print summary of matches
    for i, match_info in enumerate(matches_data):
        print(f"Match {i}: {match_info['image1_path']} -> {match_info['image2_path']}")
        print(f"Number of matches: {match_info['num_matches']}\n")
