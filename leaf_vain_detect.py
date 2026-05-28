import cv2
import numpy as np
import sys
import os
from skimage.morphology import skeletonize

def extract_veins(image_path, output_dir="vein_layers"):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define color ranges for veins
    color_ranges = {
        "green_veins":  (np.array([35, 40, 40]),  np.array([85, 255, 255])),
        "yellow_veins": (np.array([20, 100, 100]), np.array([35, 255, 255])),
        "brown_veins":  (np.array([10, 100, 20]),  np.array([20, 255, 200])),
        "black_veins":  (np.array([0, 0, 0]),      np.array([180, 255, 50]))
    }

    os.makedirs(output_dir, exist_ok=True)

    vein_stats = {}
    total_pixels = img.shape[0] * img.shape[1]

    for name, (lower, upper) in color_ranges.items():
        # Mask by color
        mask = cv2.inRange(hsv, lower, upper)

        # Emphasize thin structures (veins)
        edges = cv2.Canny(mask, 50, 150)
        veins = cv2.bitwise_and(mask, edges)

        # Skeletonize for clean vein lines
        veins_bin = veins > 0
        veins_skel = skeletonize(veins_bin).astype(np.uint8) * 255

        # Apply mask to original image
        segmented = cv2.bitwise_and(img, img, mask=veins_skel)

        # Save each vein layer
        cv2.imwrite(os.path.join(output_dir, f"{name}.png"), segmented)

        # Calculate percentage area
        vein_pixels = np.count_nonzero(veins_skel)
        percent_area = (vein_pixels / total_pixels) * 100
        vein_stats[name] = percent_area

        # Show each layer
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, segmented)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Print summary report
    print("\n--- Vein Color Summary ---")
    for color, pct in vein_stats.items():
        print(f"{color}: {pct:.2f}% of leaf area")

    return vein_stats

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_veins.py <path_to_leaf_photo>")
    else:
        extract_veins(sys.argv[1])
