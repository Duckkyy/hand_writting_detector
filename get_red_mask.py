import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_red_mask(img, save_result=False):
    """
    Given an image path, extract red colored markings from the image.

    Args:
        image_path (str): Path to the input image.
    Returns:
        out_img (numpy.ndarray): Image with only red colored markings. 
    """   
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape

    # Convert to float for easier calculations
    rgbf = rgb.astype(np.float32) / 255.0
    R, G, B = rgbf[:,:,0], rgbf[:,:,1], rgbf[:,:,2]

    # ---- Logic to detect "color" ----
    # Calculate mean and standard deviation of R,G,B channels
    mean = (R + G + B) / 3.0
    std = np.sqrt(((R - mean)**2 + (G - mean)**2 + (B - mean)**2) / 3.0)

    # If std is high => pixel has color deviation (not gray)
    # If R >> G,B => pixel has red tint
    colored_mask = (std > 0.05) | (R > 1.2 * G) | (R > 1.2 * B)

    # ---- Export result image ----
    
    result = np.zeros_like(rgb)
    result[colored_mask] = rgb[colored_mask]   # keep colored pixels        
    white_bg = np.ones_like(rgb) * 255
    out_img = np.where(colored_mask[:,:,None], rgb, white_bg).astype(np.uint8)

    if save_result:
        save_mask_path = "preprocessing/colored_mask.png"
        save_colored_path = "preprocessing/colored_pixels.png"
        os.makedirs("preprocessing", exist_ok=True)
        cv2.imwrite(save_mask_path, (colored_mask.astype(np.uint8) * 255))
        cv2.imwrite(save_colored_path, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
        print("Saved to: preprocessing/colored_mask.png  and  preprocessing/colored_pixels.png")
    else:
        save_mask_path = None
        save_colored_path = None

    return out_img, colored_mask, save_mask_path

def main(image_path):
    # Load image
    img = cv2.imread(image_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out_img, colored_mask, _ = get_red_mask(img)

    mask_uint8 = (colored_mask * 255).astype(np.uint8)

    # ---- Display ----
    plt.figure(figsize=(12,5))
    plt.subplot(1,3,1); plt.imshow(rgb); plt.title("Original"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(mask_uint8, cmap="gray"); plt.title("Mask (colored pixels)"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(out_img); plt.title("Extracted color ink"); plt.axis("off")
    plt.show()

    # ---- Save images ----
    cv2.imwrite("res1.png", mask_uint8)
    cv2.imwrite("res2.png", cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))

    print("Saved to: /mnt/data/colored_mask.png  and  /mnt/data/colored_pixels.png")

if __name__ == "__main__":
    image_path = "crops/number/image_det2_conf0.69.png"
    out_img = main(image_path)