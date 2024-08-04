import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F


class SegmentationAugmentation:
    def __init__(self, size=(256, 256)):
        self.size = size

    def __call__(self, image, mask):
        # Resize
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
    # Convert to tensor and normalize
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return normalize(to_tensor(image))

def preprocess_mask(mask):
    """Preprocess the mask."""
    # Convert to tensor
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
    # Create directories if they don't exist
    os.makedirs("data/augment_images", exist_ok=True)
    os.makedirs("data/augment_masks", exist_ok=True)

    # Get original filenames
    image_filename = os.path.basename(original_image_path)
    mask_filename = os.path.basename(original_mask_path)

    # Create new filenames
    aug_image_filename = f"aug_{image_filename}"
    aug_mask_filename = f"aug_{mask_filename}"

    # Save augmented image and mask
    image.save(os.path.join("data/augment_images", aug_image_filename))
    mask.save(os.path.join("data/augment_masks", aug_mask_filename))

# Example usage
if __name__ == "__main__":
    image_path = "../data/train_images"
    mask_path = "../data/train_masks"
    
    augmented_image, augmented_mask = load_and_preprocess(image_path, mask_path, save_augmented=True)
    print("Augmented image shape:", augmented_image.shape)
    print("Augmented mask shape:", augmented_mask.shape)
    print("Augmented files saved in data/augment_images and data/augment_masks")