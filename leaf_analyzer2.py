import cv2
import numpy as np
import sys

def analyze_leaf(image_path):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print("Could not load image. Check the path.")
        return
    
    # --- AUTOMATIC LEAF SEGMENTATION USING HSV ---
    # Convert from BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define HSV boundaries to capture greens, yellow-greens, and pure yellows
    # OpenCV Hue range is 0-180 (corresponds to 0-360 degrees)
    # Hue: 20 is deep yellow/orange, 85 is deep forest green
    lower_plant_color = np.array([20,  40,  40])   
    upper_plant_color = np.array([85, 255, 255])  
    
    # Create the binary mask (255 where leaf is, 0 everywhere else)
    leaf_mask = cv2.inRange(hsv, lower_plant_color, upper_plant_color)
    
    # Clean up the mask using Morphological Operations (removes tiny noise dots)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel) # fills tiny holes inside leaf
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)  # removes tiny stray background speckles
    
    # Count how many leaf pixels we found
    leaf_pixel_count = cv2.countNonZero(leaf_mask)
    if leaf_pixel_count < 100:  # Safeguard if no leaf matches our color boundaries
        print("Error: Could not automatically detect a green or yellow leaf in this image.")
        return
        
    # --- LAB ANALYSIS ONLY ON THE MASKED PIXELS ---
    # Convert original BGR image to LAB space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_float = lab.astype(np.float32)
    
    # Extract channels
    L = lab_float[:,:,0]
    a = lab_float[:,:,1]
    b = lab_float[:,:,2]
    
    # Compute mean values ONLY where leaf_mask is active (greater than 0)
    mean_L = np.mean(L[leaf_mask > 0])
    mean_a = np.mean(a[leaf_mask > 0])
    mean_b = np.mean(b[leaf_mask > 0])
    
    # Scale back to standard CIELAB units
    mean_L_actual = mean_L * (100.0 / 255.0)
    mean_a_actual = mean_a - 128
    mean_b_actual = mean_b - 128
    
    print(f"Leaf Pixel Coverage: {leaf_pixel_count} pixels analyzed.")
    print(f"Actual L* value: {mean_L_actual:.2f} (0 = black, 100 = white)")
    print(f"Actual a* value: {mean_a_actual:.2f} (negative = green, positive = red)")
    print(f"Actual b* value: {mean_b_actual:.2f} (negative = blue, positive = yellow)\n")
    
    # --- DECISION LOGIC ---
    if mean_a_actual < -10:
        if mean_L_actual > 55:
            print("Interpretation: Healthy, YOUNG leaf (lighter green vibrant tissue).")
        else:
            print("Interpretation: Healthy, MATURE leaf (darker green, rich chlorophyll).")
    elif mean_b_actual > 25:
        if mean_L_actual > 60:
            print("Interpretation: Chlorosis / Nutrient Deficiency (Nitrogen/Magnesium). Leaf tissue is faded and pale.")
        else:
            print("Interpretation: Withered / Wilted tissue. Leaf is losing green pigmentation and darkening toward brown/yellow.")
    elif mean_a_actual > -2 and mean_b_actual > 10:
        print("Interpretation: Severely degraded or dead leaf tissue (Brown/Necrotic).")
    else:
        print("Interpretation: Mixed signals or transitional state.")
        
    # Visual Output for Debugging: Cut out the background using our mask
    segmented_result = cv2.bitwise_and(img, img, mask=leaf_mask)
    
    # Resize for display convenience if photos are huge
    display_img = cv2.resize(segmented_result, (600, 400))
    cv2.imshow('Automated Segmentation (Only Leaf Retained)', display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python leaf_analyzer.py <path_to_leaf_photo>")
    else:
        analyze_leaf(sys.argv[1])
