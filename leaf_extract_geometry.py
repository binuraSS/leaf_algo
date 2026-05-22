import cv2
import numpy as np
import sys
import math

def extract_geometry(leaf_mask):
    """
    Finds the largest contour in the mask and extracts geometric metrics.
    Returns a dictionary of metrics and the contour itself for drawing.
    """
    contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    largest_contour = max(contours, key=cv2.contourArea)
    true_area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, closed=True)
    
    if true_area < 100:
        return None, None

    _, (width, height), _ = cv2.minAreaRect(largest_contour)
    if width == 0 or height == 0:
        return None, None
    
    length = max(width, height)
    short_side = min(width, height)
    aspect_ratio = short_side / length

    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    solidity = true_area / hull_area if hull_area > 0 else 0

    serration_index = perimeter / (2 * math.sqrt(math.pi * true_area)) if true_area > 0 else 0

    moments = cv2.moments(largest_contour)
    hu_moments = cv2.HuMoments(moments)
    
    log_hu = []
    for i in range(7):
        if hu_moments[i][0] != 0:
            log_hu.append(-1 * math.copysign(1.0, hu_moments[i][0]) * math.log10(abs(hu_moments[i][0])))
        else:
            log_hu.append(0.0)

    metrics = {
        "True Area (px)": int(true_area),
        "Aspect Ratio": round(aspect_ratio, 3),
        "Solidity": round(solidity, 3),
        "Serration Index": round(serration_index, 3),
        "Hu Moment 1 (Primary Shape)": round(log_hu[0], 3),
        "Hu Moment 2 (Asymmetry)": round(log_hu[1], 3)
    }
    
    return metrics, largest_contour

def analyze_leaf(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Could not load image. Check the path.")
        return
    
    # --- AUTOMATIC LEAF SEGMENTATION USING HSV ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_plant_color = np.array([20,  40,  40])   
    upper_plant_color = np.array([85, 255, 255])  
    
    leaf_mask = cv2.inRange(hsv, lower_plant_color, upper_plant_color)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)
    
    # --- GEOMETRIC SHAPE ANALYSIS ---
    shape_metrics, leaf_contour = extract_geometry(leaf_mask)
    
    if shape_metrics is None:
        print("Error: Could not extract reliable leaf geometry.")
        return

    print("=== EXTRACTED GEOMETRIC FEATURES ===")
    for metric, value in shape_metrics.items():
        print(f"{metric}: {value}")
    print("====================================\n")

    # --- VISUAL OUTPUT GENERATION ---
    mask_bgr = cv2.cvtColor(leaf_mask, cv2.COLOR_GRAY2BGR)
    segmented_result = cv2.bitwise_and(img, img, mask=leaf_mask)
    
    if leaf_contour is not None:
        cv2.drawContours(segmented_result, [leaf_contour], -1, (255, 0, 0), 3)

    display_w, display_h = 400, 300
    panel_orig = cv2.resize(img, (display_w, display_h))
    panel_mask = cv2.resize(mask_bgr, (display_w, display_h))
    panel_result = cv2.resize(segmented_result, (display_w, display_h))
    
    composite_display = np.hstack((panel_orig, panel_mask, panel_result))
    
    window_name = 'Diagnostic View: Original | Binary Mask | Isolated Leaf'
    cv2.imshow(window_name, composite_display)
    
    print("Press 'ESC' or focus the terminal and hit 'Ctrl+C' to close and quit.")
    
    # --- FIXED EVENT LOOP FOR HANDLING TERMINAL INTERRUPTS ---
    while True:
        # Step the waitKey loop in tiny 50ms intervals to keep the thread responsive to terminal events
        key = cv2.waitKey(50) & 0xFF
        
        # 27 is the ASCII code for the Escape Key
        if key == 27:
            break
            
        # Check if the user clicked the 'X' close window button manually
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
            
    cv2.destroyAllWindows()    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python leaf_analyzer.py <path_to_leaf_photo>")
    else:
        # Wrap execution in a clean Python try/except block to catch Ctrl+C signals
        try:
            analyze_leaf(sys.argv[1])
        except KeyboardInterrupt:
            print("\nProgram execution halted via Ctrl+C. Cleaning up windows and exiting.")
            cv2.destroyAllWindows()
            sys.exit(0)