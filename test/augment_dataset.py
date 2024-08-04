import os
from ..app.preprocessing import load_and_preprocess


def augment_dataset(image_dir, mask_dir, num_augmentations=5):
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        mask_path = os.path.join(mask_dir, image_file.replace('.jpg', '.png'))  # Adjust if your mask files have a different extension
        
        for i in range(num_augmentations):
            load_and_preprocess(image_path, mask_path, augment=True, save_augmented=True)
    
    print(f"Augmentation complete. {len(image_files) * num_augmentations} new images and masks created.")

# Example usage
image_dir = "../data/train_images"
mask_dir = "../data/train_masks"
augment_dataset(image_dir, mask_dir)