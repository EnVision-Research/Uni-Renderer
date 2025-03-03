import os
from PIL import Image
import cv2
import numpy as np
# Define the paths to your folders
images_folder = '/hpc2hdd/home/zchen379/sd3/diff_rendering_unet/dataset/relight/rgb_gt'
masks_folder = '/hpc2hdd/home/zchen379/sd3/diff_rendering_unet/dataset/relight/masks'
output_folder = '/hpc2hdd/home/zchen379/sd3/diff_rendering_unet/dataset/relight/without_bg_gt'


# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Get a list of image filenames (assuming they start with 'rgb_image_' and end with '.png')
image_filenames = [f for f in os.listdir(images_folder) if f.lower().startswith('rgb_gt_') and f.lower().endswith('.png')]

for image_filename in image_filenames:
    # Extract the common identifier
    identifier = image_filename[len('rgb_gt_'):]  # Remove 'rgb_image_' prefix

    # Construct the corresponding mask filename
    mask_filename = f'mask_image_{identifier}'

    image_path = os.path.join(images_folder, image_filename)
    mask_path = os.path.join(masks_folder, mask_filename)
    output_path = os.path.join(output_folder, f'{identifier}')  # Save as 'xx_0.png'

    # Check if the corresponding mask exists
    if not os.path.exists(mask_path):
        print(f"No mask found for {image_filename}, skipping.")
        continue

    # Load the image and mask
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')  # Convert mask to grayscale

    # Convert images to numpy arrays
    image_array = np.array(image).astype(np.float32)
    mask_array = np.array(mask).astype(np.float32) / 255.0  # Normalize mask to [0, 1]

    # Ensure mask is binary (0 or 1)
    mask_array = (mask_array > 0.5).astype(np.float32)

    # Create the inverse mask
    inv_mask_array = 1.0 - mask_array

    # Multiply image and mask, set background to white
    result_array = image_array * mask_array[:, :, np.newaxis] + 255 * inv_mask_array[:, :, np.newaxis]

    # Convert result back to uint8 and clip values
    result_array = np.clip(result_array, 0, 255).astype(np.uint8)

    # Convert result back to image
    result_image = Image.fromarray(result_array)

    # Save the result
    result_image.save(output_path)

    print(f"Processed {image_filename}")