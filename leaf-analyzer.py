import cv2
import numpy as np
import sys

def analyze_leaf(image_path):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print("Could not load image. Check the path.")
        return
    
    # Convert from BGR (OpenCV default) to LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # You may want to roughly isolate the leaf by thresholding on greenness
    # For simplicity, we'll analyze the whole image – but better to crop manually first
    # Let's assume you've cropped the photo to mostly the leaf (using any photo editor)
    
    # Convert to float for easier math
    lab_float = lab.astype(np.float32)
    
    # Extract L, a, b channels
    L = lab_float[:,:,0]
    a = lab_float[:,:,1]
    b = lab_float[:,:,2]
    
    # Compute mean values (only over pixels that are not too dark/bright? skip for now)
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    mean_a_actual = mean_a - 128
    mean_b_actual = mean_b - 128
    
    print(f"Actual  a* value: {mean_a_actual:.2f} (negative = green, positive = red)")
    print(f"Actual  b* value: {mean_b_actual:.2f} (negative = blue, positive = yellow)")
    
    # Simple rule-based interpretation (you will refine with your own plant data)
    if mean_a_actual < -15 and mean_b_actual < 20:
        print("Interpretation: Likely a mature, healthy green leaf.")
    elif mean_a_actual < -5 and mean_b_actual > 20:
        # Leaf 1 would fall here (-19.75, 24.32)
        print("Interpretation: Possibly young leaf or early nutrient deficiency (yellowish green).")
    elif mean_a_actual > -5 and mean_b_actual > 30:
        # Leaf 2 would fall here (-6.02, 35.58)
        print("Interpretation: Possible magnesium deficiency or other yellowing stress.")
    else:
        print("Interpretation: Mixed signals. Compare with known healthy leaf baseline.")
        
    # After converting to LAB, add these lines:
    lab_display = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # back to BGR for display
    cv2.imshow('Leaf in LAB view', lab_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python leaf_analyzer.py <path_to_leaf_photo>")
    else:
        analyze_leaf(sys.argv[1])
