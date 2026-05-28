import cv2
import numpy as np
import sys  # 1. Import sys to read command line arguments

def analyze_leaf_deficiency(image_path):
    # Load the color image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load the leaf image from path: '{image_path}'")
        return

    # Resize slightly for consistent structural operations
    img = cv2.resize(img, (600, 600))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Segment the Leaf from the background 
    _, leaf_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Fill any holes inside the leaf body
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel_close)

    # Dissect Veins using Morphological Black-Hat 
    kernel_vein = cv2.getStructuringElement(cv2.MORPH_CROSS, (15, 15))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_vein)
    
    # Threshold the structural skeleton to form a binary Vein Mask
    _, vein_mask_raw = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    vein_mask = cv2.bitwise_and(vein_mask_raw, leaf_mask) 

    # Dissect Lamina (Leaf body minus the veins)
    lamina_mask = cv2.bitwise_and(leaf_mask, cv2.bitwise_not(vein_mask))

    # Define HSV thresholds for Green and Yellow
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    lower_yellow = np.array([15, 40, 40])
    upper_yellow = np.array([35, 255, 255])

    # Generate color maps for the whole image
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Analyze Vein Color Composition
    veins_green = cv2.bitwise_and(vein_mask, green_mask)
    veins_yellow = cv2.bitwise_and(vein_mask, yellow_mask)
    
    vein_total_pixels = np.sum(vein_mask == 255)
    vein_green_pct = (np.sum(veins_green == 255) / vein_total_pixels) * 100 if vein_total_pixels > 0 else 0
    vein_yellow_pct = (np.sum(veins_yellow == 255) / vein_total_pixels) * 100 if vein_total_pixels > 0 else 0

    # Analyze Lamina Color Composition
    lamina_green = cv2.bitwise_and(lamina_mask, green_mask)
    lamina_yellow = cv2.bitwise_and(lamina_mask, yellow_mask)
    
    lamina_total_pixels = np.sum(lamina_mask == 255)
    lamina_green_pct = (np.sum(lamina_green == 255) / lamina_total_pixels) * 100 if lamina_total_pixels > 0 else 0
    lamina_yellow_pct = (np.sum(lamina_yellow == 255) / lamina_total_pixels) * 100 if lamina_total_pixels > 0 else 0

    # Classification Logic
    print("\n====== DIAGNOSTIC REPORT ======")
    print(f"Veins color composition : Green: {vein_green_pct:.1f}%, Yellow: {vein_yellow_pct:.1f}%")
    print(f"Lamina color composition: Green: {lamina_green_pct:.1f}%, Yellow: {lamina_yellow_pct:.1f}%")
    print("-------------------------------")

    if vein_green_pct > 50 and lamina_yellow_pct > 40:
        diagnosis = "MAGNESIUM (Mg) DEFICIENCY DETECTED\n(Interveinal Chlorosis: Veins remain green, lamina yellows)"
    elif vein_yellow_pct > 40 and lamina_yellow_pct > 40:
        diagnosis = "NITROGEN (N) DEFICIENCY DETECTED\n(Uniform Chlorosis: Both veins and lamina turning yellow)"
    else:
        diagnosis = "Healthy or Alternative Condition (Inconclusive structural metrics)"
    
    print(f"Result: {diagnosis}")

    # Visualizing the breakdown
    vein_visualization = np.zeros_like(img)
    vein_visualization[vein_mask == 255] = [0, 0, 255]  # Red overlay for veins
    
    lamina_visualization = cv2.bitwise_and(img, img, mask=lamina_mask)

    cv2.imshow("Original Leaf", img)
    cv2.imshow("Isolated Vein Skeleton (Red Mapping)", vein_visualization)
    cv2.imshow("Isolated Lamina Body", lamina_visualization)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 2. Add entry point execution block to verify argument inputs
if __name__ == "__main__":
    # Check if user forgot to provide a filename
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <path_to_leaf_image>")
        sys.exit(1)
        
    # Pick the first argument passed after the script name
    input_image_path = sys.argv[1]
    analyze_leaf_deficiency(input_image_path)
