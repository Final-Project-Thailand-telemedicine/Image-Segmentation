import os
import random
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F

def get_data_dir():
    data_dir = os.environ.get('IMAGE_SEGMENTATION_DATA_DIR')
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    return data_dir

class SegmentationAugmentation:
    def __init__(self, size=(512, 512)):  # Changed default size to 512x512
        self.size = size

    def __call__(self, image, mask):
        # Ensure image and mask are the correct size
        image = F.resize(image, self.size)
        mask = F.resize(mask, self.size, interpolation=transforms.InterpolationMode.NEAREST)

        # Random horizontal flipping
        if torch.rand(1) > 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)

        # Random vertical flipping
        if torch.rand(1) > 0.5:
            image = F.vflip(image)
            mask = F.vflip(mask)

        # Random rotation
        if torch.rand(1) > 0.5:
            angle = torch.randint(-30, 30, (1,)).item()
            image = F.rotate(image, angle)
            mask = F.rotate(mask, angle, interpolation=transforms.InterpolationMode.NEAREST)

        # Random brightness and contrast
        if torch.rand(1) > 0.5:
            brightness_factor = torch.FloatTensor(1).uniform_(0.8, 1.2).item()
            contrast_factor = torch.FloatTensor(1).uniform_(0.8, 1.2).item()
            image = F.adjust_brightness(image, brightness_factor)
            image = F.adjust_contrast(image, contrast_factor)

        return image, mask

def preprocess_image(image):
    """Preprocess the input image."""
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return normalize(to_tensor(image))

def preprocess_mask(mask):
    """Preprocess the mask."""
    to_tensor = transforms.ToTensor()
    return to_tensor(mask)

def load_and_preprocess(image_path, mask_path, augment=True, save_augmented=False):
    """Load, preprocess, and augment an image and its mask."""
    # Load image and mask
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')

    # Augment
    if augment:
        augmentation = SegmentationAugmentation()
        image, mask = augmentation(image, mask)

    # Save augmented images if requested
    if save_augmented:
        save_augmented_files(image, mask, image_path, mask_path)

    # Preprocess
    image = preprocess_image(image)
    mask = preprocess_mask(mask)

    return image, mask

def save_augmented_files(image, mask, original_image_path, original_mask_path):
    """Save augmented image and mask to specified directories."""
    data_dir = get_data_dir()
    
    # Create directories if they don't exist
    os.makedirs(os.path.join(data_dir, "augment_images"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "augment_masks"), exist_ok=True)

    # Get original filenames
    image_filename = os.path.basename(original_image_path)
    mask_filename = os.path.basename(original_mask_path)

    # Create new filenames
    aug_image_filename = f"aug_{image_filename}"
    aug_mask_filename = f"aug_{mask_filename}"

    # Save augmented image and mask
    image.save(os.path.join(data_dir, "augment_images", aug_image_filename))
    mask.save(os.path.join(data_dir, "augment_masks", aug_mask_filename))

def augment_dataset(source_image_dir, source_mask_dir, target_total=10000, augmentations_per_image=3):
    """Augment the entire dataset to reach the target total number of images."""
    data_dir = get_data_dir()
    augmented_image_dir = os.path.join(data_dir, "augment_images")
    augmented_mask_dir = os.path.join(data_dir, "augment_masks")
    
    os.makedirs(augmented_image_dir, exist_ok=True)
    os.makedirs(augmented_mask_dir, exist_ok=True)

    image_files = [f for f in os.listdir(source_image_dir) if f.endswith('.png')]
    total_original = len(image_files)
    total_augmented = 0
    
    print(f"Found {total_original} original images. Target total: {target_total}")

    while total_original + total_augmented < target_total:
        for image_file in image_files:
            if total_original + total_augmented >= target_total:
                break

            mask_file = image_file  # Assuming mask has the same filename as the image
            image_path = os.path.join(source_image_dir, image_file)
            mask_path = os.path.join(source_mask_dir, mask_file)

            if not os.path.exists(mask_path):
                print(f"Corresponding mask file not found for image {image_file}. Skipping...")
                continue

            for _ in range(augmentations_per_image):
                if total_original + total_augmented >= target_total:
                    break

                try:
                    augmented_image, augmented_mask = load_and_preprocess(image_path, mask_path, save_augmented=True)
                    total_augmented += 1
                    print(f"Augmented image {total_augmented}: {image_file}")
                except Exception as e:
                    print(f"Error processing {image_file}: {str(e)}")
                    continue

    print(f"Augmentation complete. Total images: {total_original + total_augmented}")
    print(f"Original images: {total_original}")
    print(f"Augmented images: {total_augmented}")

# Example usage
if __name__ == "__main__":
    data_dir = get_data_dir()
    source_image_dir = os.path.join(data_dir, "train_images")
    source_mask_dir = os.path.join(data_dir, "train_masks")
    
    if not os.path.exists(source_image_dir) or not os.path.exists(source_mask_dir):
        raise FileNotFoundError(f"Data directory not found. Please set the IMAGE_SEGMENTATION_DATA_DIR environment variable or ensure the data is in {data_dir}")

    augment_dataset(source_image_dir, source_mask_dir, target_total=10000, augmentations_per_image=3)
    data_dir = get_data_dir()
    image_path = os.path.join(data_dir, "train_images")
    mask_path = os.path.join(data_dir, "train_masks")
    
    if not os.path.exists(image_path) or not os.path.exists(mask_path):
        raise FileNotFoundError(f"Data directory not found. Please set the IMAGE_SEGMENTATION_DATA_DIR environment variable or ensure the data is in {data_dir}")
    
    image_files = [f for f in os.listdir(image_path) if f.endswith('.png')]
    
    if not image_files:
        raise FileNotFoundError(f"No PNG image files found in {image_path}")
    
    num_augmentations = 1  # Change this if you want to perform multiple augmentations

    for i in range(num_augmentations):
        image_file = random.choice(image_files)
        mask_file = image_file  # Assuming mask has the same filename as the image
        
        image_file_path = os.path.join(image_path, image_file)
        mask_file_path = os.path.join(mask_path, mask_file)
        
        if not os.path.exists(mask_file_path):
            print(f"Corresponding mask file not found for image {image_file}. Skipping...")
            continue
        
        print(f"Processing image: {image_file} and its corresponding mask")
        
        try:
            augmented_image, augmented_mask = load_and_preprocess(image_file_path, mask_file_path, save_augmented=True)
            print(f"Augmentation {i+1}:")
            print("Augmented image shape:", augmented_image.shape)
            print("Augmented mask shape:", augmented_mask.shape)
            print(f"Augmented files saved in {os.path.join(data_dir, 'augment_images')} and {os.path.join(data_dir, 'augment_masks')}")
            print()
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            continue

    print("Augmentation process completed.")