import cv2
import numpy as np
import sys
import os

def get_min_max_intensity(hsv_img, structural_mask, color_mask):
    """Calculates min and max brightness (V channel) where structure and color overlap."""
    # Combine the structural mask (vein/lamina) with the color mask (green/yellow)
    combined_mask = cv2.bitwise_and(structural_mask, color_mask)
    
    # Extract only the 'Value' (brightness) channel for these specific pixels
    v_channel = hsv_img[:, :, 2]
    pixels = v_channel[combined_mask == 255]
    
    if len(pixels) > 0:
        return int(np.min(pixels)), int(np.max(pixels))
    return 0, 0  # Return 0s if no pixels match this color

def analyze_leaf_deficiency(image_path, report_file):
    img = cv2.imread(image_path)
    if img is None:
        error_msg = f"Error: Could not load the leaf image from path: '{image_path}'\n"
        print(error_msg)
        report_file.write(error_msg)
        return

    img = cv2.resize(img, (600, 600))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Segment Leaf
    _, leaf_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel_close)

    # Dissect Veins 
    kernel_vein = cv2.getStructuringElement(cv2.MORPH_CROSS, (15, 15))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_vein)
    _, vein_mask_raw = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    vein_mask = cv2.bitwise_and(vein_mask_raw, leaf_mask) 

    # Dissect Lamina
    lamina_mask = cv2.bitwise_and(leaf_mask, cv2.bitwise_not(vein_mask))

    # HSV thresholds
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    lower_yellow = np.array([15, 40, 40])
    upper_yellow = np.array([34, 255, 255])

    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Area Percentage Calculations
    vein_total_pixels = np.sum(vein_mask == 255)
    vein_green_pct = (np.sum(cv2.bitwise_and(vein_mask, green_mask) == 255) / vein_total_pixels) * 100 if vein_total_pixels > 0 else 0
    vein_yellow_pct = (np.sum(cv2.bitwise_and(vein_mask, yellow_mask) == 255) / vein_total_pixels) * 100 if vein_total_pixels > 0 else 0

    lamina_total_pixels = np.sum(lamina_mask == 255)
    lamina_green_pct = (np.sum(cv2.bitwise_and(lamina_mask, green_mask) == 255) / lamina_total_pixels) * 100 if lamina_total_pixels > 0 else 0
    lamina_yellow_pct = (np.sum(cv2.bitwise_and(lamina_mask, yellow_mask) == 255) / lamina_total_pixels) * 100 if lamina_total_pixels > 0 else 0

    # 🛠️ NEW: Calculate Min and Max Intensity ranges
    v_g_min, v_g_max = get_min_max_intensity(hsv, vein_mask, green_mask)
    v_y_min, v_y_max = get_min_max_intensity(hsv, vein_mask, yellow_mask)
    
    l_g_min, l_g_max = get_min_max_intensity(hsv, lamina_mask, green_mask)
    l_y_min, l_y_max = get_min_max_intensity(hsv, lamina_mask, yellow_mask)

    # Classification Logic
    if vein_green_pct > 50 and lamina_yellow_pct > 40:
        diagnosis = "MAGNESIUM (Mg) DEFICIENCY DETECTED (Interveinal Chlorosis)"
    elif vein_yellow_pct > 40 and lamina_yellow_pct > 40:
        diagnosis = "NITROGEN (N) DEFICIENCY DETECTED (Uniform Chlorosis)"
    else:
        diagnosis = "Healthy or Alternative Condition (Inconclusive metrics)"
    
    # Updated text formatting with Min/Max metrics
    report_text = (
        f"\nIMAGE FILE: {os.path.basename(image_path)}\n"
        f"--------------------------------------------------------------------------------\n"
        f"VEINS  | Green Area: {vein_green_pct:.1f}% [Intensity Min: {v_g_min}, Max: {v_g_max}]\n"
        f"       | Yellow Area: {vein_yellow_pct:.1f}% [Intensity Min: {v_y_min}, Max: {v_y_max}]\n"
        f"--------------------------------------------------------------------------------\n"
        f"LAMINA | Green Area: {lamina_green_pct:.1f}% [Intensity Min: {l_g_min}, Max: {l_g_max}]\n"
        f"       | Yellow Area: {lamina_yellow_pct:.1f}% [Intensity Min: {l_y_min}, Max: {l_y_max}]\n"
        f"--------------------------------------------------------------------------------\n"
        f"Result : {diagnosis}\n"
        f"--------------------------------------------------------------------------------\n"
    )
    
    print(report_text)
    report_file.write(report_text) 

    # Visualization Windows
    vein_visualization = np.zeros_like(img)
    vein_visualization[vein_mask == 255] = [0, 0, 255] 
    lamina_visualization = cv2.bitwise_and(img, img, mask=lamina_mask)

    cv2.imshow("Original Leaf Image", img)
    cv2.imshow("Isolated Vein Skeleton", vein_visualization)
    cv2.imshow("Isolated Lamina Body", lamina_visualization)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)  

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <path_to_leaf1> ...")
        sys.exit(1)
        
    report_filename = "leaf_diagnostic_report.txt"
    
    with open(report_filename, "w") as f:
        f.write("================================================================================\n")
        f.write("                          PLANT DEFICIENCY ANALYSIS REPORT                      \n")
        f.write("================================================================================\n")
        
        for i in range(1, len(sys.argv)):
            analyze_leaf_deficiency(sys.argv[i], f)
            
    print(f"\nProcessing complete. All results saved safely to '{report_filename}'.")
