import cv2
import os
import numpy as np

def preprocess_images(input_folder, output_folder):
    """Enhances images using CLAHE, Gamma Correction, and Edge Detection."""
    
    os.makedirs(output_folder, exist_ok=True)
    
    # List images
    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".png")])

    for i, image_file in enumerate(image_files):
        img_path = os.path.join(input_folder, image_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Error: Unable to load {image_file}")
            continue

        # Increase Resolution
        scale_factor = 2  
        img_high_res = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        # Apply Gamma Correction
        gamma = 1.5  
        img_gamma_corrected = np.power(img_high_res / 255.0, gamma) * 255.0
        img_gamma_corrected = img_gamma_corrected.astype(np.uint8)

        # Enhance Contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32, 32))
        img_enhanced = clahe.apply(img_gamma_corrected)

        # Apply Edge Detection
        edges = cv2.Canny(img_enhanced, threshold1=50, threshold2=150)

        # Combine Edges with Enhanced Image
        img_combined = cv2.addWeighted(img_enhanced, 0.8, edges, 0.2, 0)

        # Save processed image
        output_path = os.path.join(output_folder, f"processed_{image_file}")
        cv2.imwrite(output_path, img_combined)

        print(f"Processed {i+1}/{len(image_files)}: {image_file}")

    print("Preprocessing complete. Processed images saved!")

