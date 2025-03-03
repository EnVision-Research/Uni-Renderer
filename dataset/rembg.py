import cv2
import numpy as np

# Load the RGB image and mask
rgb = cv2.imread('rgb_image.png')   # Replace with your RGB image filename
mask = cv2.imread('mask_image.png', cv2.IMREAD_GRAYSCALE)  # Load mask as grayscale

# Ensure the mask is binary (0 or 255)
_, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# If the mask is single-channel and the image is three-channel, replicate the mask to all channels
if len(mask_binary.shape) == 2 and len(rgb.shape) == 3 and rgb.shape[2] == 3:
    mask_rgb = cv2.merge([mask_binary, mask_binary, mask_binary])
else:
    mask_rgb = mask_binary

# Apply the mask to the image
masked_image = cv2.bitwise_and(rgb, mask_rgb)

# Save the masked image
cv2.imwrite('masked_image.png', masked_image)
