import cv2
import numpy as np
import sys
import os

def analyze_leaf_deficiency(image_path, report_file):
    # 1. Load the color image
    img = cv2.imread(image_path)
    if img is None:
        error_msg = f"Error: Could not load the leaf image from path: '{image_path}'\n"
        print(error_msg)
        report_file.write(error_msg)
        return

    # Resize for consistent structural operations
    img = cv2.resize(img, (600, 600))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Segment the Leaf from the background 
    _, leaf_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel_close)

    # 3. Dissect Veins using Morphological Black-Hat 
    kernel_vein = cv2.getStructuringElement(cv2.MORPH_CROSS, (15, 15))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_vein)
    _, vein_mask_raw = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    vein_mask = cv2.bitwise_and(vein_mask_raw, leaf_mask) 

    # 4. Dissect Lamina (Leaf body minus the veins)
    lamina_mask = cv2.bitwise_and(leaf_mask, cv2.bitwise_not(vein_mask))

    # 5. Define HSV thresholds for Green and Yellow
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    lower_yellow = np.array([15, 40, 40])
    upper_yellow = np.array([35, 255, 255])

    # Generate color maps
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 6. Analyze Vein Color Composition
    veins_green = cv2.bitwise_and(vein_mask, green_mask)
    veins_yellow = cv2.bitwise_and(vein_mask, yellow_mask)
    vein_total_pixels = np.sum(vein_mask == 255)
    vein_green_pct = (np.sum(veins_green == 255) / vein_total_pixels) * 100 if vein_total_pixels > 0 else 0
    vein_yellow_pct = (np.sum(veins_yellow == 255) / vein_total_pixels) * 100 if vein_total_pixels > 0 else 0

    # 7. Analyze Lamina Color Composition
    lamina_green = cv2.bitwise_and(lamina_mask, green_mask)
    lamina_yellow = cv2.bitwise_and(lamina_mask, yellow_mask)
    lamina_total_pixels = np.sum(lamina_mask == 255)
    lamina_green_pct = (np.sum(lamina_green == 255) / lamina_total_pixels) * 100 if lamina_total_pixels > 0 else 0
    lamina_yellow_pct = (np.sum(lamina_yellow == 255) / lamina_total_pixels) * 100 if lamina_total_pixels > 0 else 0

    # 8. Classification Logic
    if vein_green_pct > 50 and lamina_yellow_pct > 40:
        diagnosis = "MAGNESIUM (Mg) DEFICIENCY DETECTED (Interveinal Chlorosis)"
    elif vein_yellow_pct > 40 and lamina_yellow_pct > 40:
        diagnosis = "NITROGEN (N) DEFICIENCY DETECTED (Uniform Chlorosis)"
    else:
        diagnosis = "Healthy or Alternative Condition (Inconclusive metrics)"
    
    # 9. Format outputs for terminal display
    report_text = (
        f"\nIMAGE FILE: {os.path.basename(image_path)}\n"
        f"--------------------------------------------------\n"
        f"Veins color composition : Green: {vein_green_pct:.1f}%, Yellow: {vein_yellow_pct:.1f}%\n"
        f"Lamina color composition: Green: {lamina_green_pct:.1f}%, Yellow: {lamina_yellow_pct:.1f}%\n"
        f"Result                  : {diagnosis}\n"
        f"--------------------------------------------------\n"
    )
    
    print(report_text)
    report_file.write(report_text) 

    # 10. Visualization Windows
    vein_visualization = np.zeros_like(img)
    vein_visualization[vein_mask == 255] = [0, 0, 255] # Red mapping for veins
    lamina_visualization = cv2.bitwise_and(img, img, mask=lamina_mask)

    # Use a fixed window name so it doesn't create endless new windows
    cv2.imshow("Original Leaf Image", img)
    cv2.imshow("Isolated Vein Skeleton", vein_visualization)
    cv2.imshow("Isolated Lamina Body", lamina_visualization)
    
    print("👉 CLICK on an image window and press ANY KEY to move to the next image...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # 🛠️ Crucial Fix: Clears OpenCV event buffers to allow the loop to continue

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <path_to_leaf1> <path_to_leaf2> ...")
        sys.exit(1)
        
    report_filename = "leaf_diagnostic_report.txt"
    
    with open(report_filename, "w") as f:
        f.write("==================================================\n")
        f.write("          PLANT DEFICIENCY ANALYSIS REPORT        \n")
        f.write("==================================================\n")
        
        for i in range(1, len(sys.argv)):
            image_arg = sys.argv[i]
            analyze_leaf_deficiency(image_arg, f)
            
    print(f"\nProcessing complete. All results saved safely to '{report_filename}'.")
