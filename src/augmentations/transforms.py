"""
Image augmentation transforms using Albumentations

Augmentations are applied independently to left and right patches
to create more training variety.
"""
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Optional


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(
    img_size: int = 768,
    p_hflip: float = 0.5,
    p_vflip: float = 0.5,
    p_rotate90: float = 0.5,
    use_color_jitter: bool = True,
) -> A.Compose:
    """
    Get training augmentation transforms.
    
    Strategy:
    - HorizontalFlip, VerticalFlip, RandomRotate90 for geometric variety
    - ColorJitter for color augmentation
    - Applied independently to each patch
    
    Args:
        img_size: Target image size after resize
        p_hflip: Probability of horizontal flip
        p_vflip: Probability of vertical flip
        p_rotate90: Probability of 90-degree rotation
        use_color_jitter: Whether to apply color jitter
        
    Returns:
        Albumentations Compose object
    """
    transforms_list = [
        # Resize to target size (high resolution maintained)
        A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
        
        # Geometric augmentations
        A.HorizontalFlip(p=p_hflip),
        A.VerticalFlip(p=p_vflip),
        A.RandomRotate90(p=p_rotate90),
    ]
    
    if use_color_jitter:
        transforms_list.append(
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5,
            )
        )
    
    # Normalization and tensor conversion
    transforms_list.extend([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    
    return A.Compose(transforms_list)


def get_val_transforms(img_size: int = 768) -> A.Compose:
    """
    Get validation/test transforms (no augmentation).
    
    Args:
        img_size: Target image size after resize
        
    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_tta_transforms(img_size: int = 768) -> list:
    """
    Get Test-Time Augmentation transforms.
    
    Returns list of transforms for TTA:
    1. Original (no augmentation)
    2. Horizontal flip
    3. Vertical flip
    
    Args:
        img_size: Target image size after resize
        
    Returns:
        List of Albumentations Compose objects
    """
    base = [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
    
    original = A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
        *base,
    ])
    
    hflip = A.Compose([
        A.HorizontalFlip(p=1.0),
        A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
        *base,
    ])
    
    vflip = A.Compose([
        A.VerticalFlip(p=1.0),
        A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
        *base,
    ])
    
    return [original, hflip, vflip]


def get_transforms(
    mode: str = "train",
    img_size: int = 768,
    **kwargs,
) -> A.Compose:
    """
    Factory function to get transforms.
    
    Args:
        mode: 'train', 'val', or 'test'
        img_size: Target image size
        **kwargs: Additional arguments for train transforms
        
    Returns:
        Albumentations Compose object
    """
    if mode == "train":
        return get_train_transforms(img_size, **kwargs)
    else:
        return get_val_transforms(img_size)
