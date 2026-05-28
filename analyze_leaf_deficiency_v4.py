import cv2
import numpy as np
import sys
import os

def get_min_max_intensity(hsv_img, structural_mask, color_mask):
    """Calculates min and max brightness (V channel) where structure and color overlap."""
    combined_mask = cv2.bitwise_and(structural_mask, color_mask)
    v_channel = hsv_img[:, :, 2]
    pixels = v_channel[combined_mask == 255]
    
    if len(pixels) > 0:
        return int(np.min(pixels)), int(np.max(pixels))
    return 0, 0

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

    # 1. Segment Leaf Body
    _, leaf_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel_close)

    # 2. Dissect Structural Veins and Lamina
    kernel_vein = cv2.getStructuringElement(cv2.MORPH_CROSS, (15, 15))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_vein)
    _, vein_mask_raw = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    vein_mask = cv2.bitwise_and(vein_mask_raw, leaf_mask) 
    lamina_mask = cv2.bitwise_and(leaf_mask, cv2.bitwise_not(vein_mask))

    # 3. Define Precise HSV Ranges for All Target Colors [2]
    # Format: np.array([Hue, Saturation, Value])
    
    # Standard Green (Healthy, deep greens)
    lower_green = np.array([36, 60, 40])
    upper_green = np.array([85, 255, 255])
    
    # Yellow (Chlorosis from Mg / N deficiency)
    lower_yellow = np.array([21, 60, 80])
    upper_yellow = np.array([35, 255, 255])
    
    # Pale Green (Early wilting, low nitrogen fade)
    lower_pale_green = np.array([36, 15, 100])
    upper_pale_green = np.array([85, 59, 255])
    
    # Brown / Scorched (K deficiency margin necrosis, tissue death)
    lower_brown = np.array([5, 50, 30])
    upper_brown = np.array([20, 255, 180])
    
    # Dark Purple / Reddish Purple (Phosphorus or severe stress signatures)
    lower_purple = np.array([130, 40, 20])
    upper_purple = np.array([175, 255, 150])

    # 4. Generate Color Extraction Binary Masks
    masks = {
        'Green': cv2.inRange(hsv, lower_green, upper_green),
        'Yellow': cv2.inRange(hsv, lower_yellow, upper_yellow),
        'Pale Green': cv2.inRange(hsv, lower_pale_green, upper_pale_green),
        'Brown/Scorched': cv2.inRange(hsv, lower_brown, upper_brown),
        'Dark Purple': cv2.inRange(hsv, lower_purple, upper_purple)
    }

    # 5. Compile Diagnostic Results Dataset
    vein_total_pixels = np.sum(vein_mask == 255)
    lamina_total_pixels = np.sum(lamina_mask == 255)

    report_text = f"\nIMAGE FILE: {os.path.basename(image_path)}\n"
    report_text += "="*85 + "\n"
    report_text += f"{'SEGMENT':<12} | {'COLOR':<15} | {'AREA %':<10} | {'INTENSITY RANGE (MIN - MAX)':<30}\n"
    report_text += "="*85 + "\n"

    # Process structural evaluation data
    for seg_name, seg_mask, total_pix in [('VEINS', vein_mask, vein_total_pixels), ('LAMINA', lamina_mask, lamina_total_pixels)]:
        if total_pix == 0:
            continue
        first_row = True
        for color_name, color_mask in masks.items():
            overlapping_pixels = np.sum(cv2.bitwise_and(seg_mask, color_mask) == 255)
            area_pct = (overlapping_pixels / total_pix) * 100
            c_min, c_max = get_min_max_intensity(hsv, seg_mask, color_mask)
            
            display_name = seg_name if first_row else ""
            report_text += f"{display_name:<12} | {color_name:<15} | {area_pct:>8.1f}% | [Min: {c_min:>3}, Max: {c_max:>3}]\n"
            first_row = False
        report_text += "-"*85 + "\n"

    # 6. Basic Structural Diagnosis Alerts
    # Pull percentages directly to provide quick hints
    l_yellow_pct = (np.sum(cv2.bitwise_and(lamina_mask, masks['Yellow']) == 255) / lamina_total_pixels) * 100 if lamina_total_pixels > 0 else 0
    l_brown_pct = (np.sum(cv2.bitwise_and(lamina_mask, masks['Brown/Scorched']) == 255) / lamina_total_pixels) * 100 if lamina_total_pixels > 0 else 0
    l_purple_pct = (np.sum(cv2.bitwise_and(lamina_mask, masks['Dark Purple']) == 255) / lamina_total_pixels) * 100 if lamina_total_pixels > 0 else 0
    l_pale_pct = (np.sum(cv2.bitwise_and(lamina_mask, masks['Pale Green']) == 255) / lamina_total_pixels) * 100 if lamina_total_pixels > 0 else 0

    alerts = []
    if l_brown_pct > 15: alerts.append("⚠️ HIGH NEUROTIC BROWN DETECTED: Review Potassium (K) Margin Scorch.")
    if l_purple_pct > 10: alerts.append("⚠️ PURPLE ANOMALY DETECTED: Check Phosphorus status or heavy anthocyanin stress.")
    if l_pale_pct > 30:  alerts.append("⚠️ HIGH PALE GREEN AREA: Possible early wilting or moisture transport stress.")
    if l_yellow_pct > 35: alerts.append("⚠️ HIGH YELLOW AREA: Active Chlorosis (Reference previous Mg vs N logic).")
    
    if not alerts:
        alerts.append("✅ Tissue areas align standard color composition profile bounds.")

    report_text += "DIAGNOSTIC ALERTS:\n" + "\n".join(alerts) + "\n"
    report_text += "="*85 + "\n"

    print(report_text)
    report_file.write(report_text)

    # 7. Multi-Color Target Visualization Maps
    canvas = img.copy()
    # Draw colored boundaries over the original image for visual feedback
    canvas[masks['Brown/Scorched'] == 255] = [0, 69, 139]   # Brown highlights
    canvas[masks['Dark Purple'] == 255] = [128, 0, 128]    # Purple highlights
    canvas[masks['Pale Green'] == 255] = [200, 255, 200]   # Light tint for pale green

    cv2.imshow("Original Leaf Specimen", img)
    cv2.imshow("Multi-Deficiency Color Analysis Overlays", canvas)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <leaf_image1> <leaf_image2> ...")
        sys.exit(1)
        
    report_filename = "expanded_leaf_report.txt"
    #with open(report_filename, "w") as f:
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write("="*85 + "\n")
        f.write("                     ADVANCED MULTI-NUTRIENT DEFICIENCY MATRIX                  \n")
        f.write("="*85 + "\n")
        
        for i in range(1, len(sys.argv)):
            analyze_leaf_deficiency(sys.argv[i], f)
            
    print(f"\nProcessing complete. Metrics saved to '{report_filename}'.")
