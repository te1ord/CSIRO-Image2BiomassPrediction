"""
Image augmentation transforms using Albumentations

Augmentations are applied independently to left and right patches
to create more training variety.
"""
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Optional, List



# IMAGENET_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_STD  = (0.229, 0.224, 0.225)

# # ---- Optional: for multi-channel (RGB + extra channels) ----
# class RGBOnlyTransform(A.ImageOnlyTransform):
#     """
#     Apply an Albumentations ImageOnlyTransform to the first 3 channels only.
#     Extra channels are kept unchanged.
#     Useful for JPEG / ColorJitter etc when you have extra index channels.
#     """
#     def __init__(self, inner: A.ImageOnlyTransform, n_extra_channels: int = 0, always_apply=False, p=0.5):
#         super().__init__(always_apply=always_apply, p=p)
#         self.inner = inner
#         self.n_extra = n_extra_channels

#     def apply(self, img, **params):
#         if self.n_extra <= 0 or img.shape[2] == 3:
#             return self.inner.apply(img, **params)

#         rgb = img[:, :, :3]
#         extra = img[:, :, 3:]
#         rgb_aug = self.inner.apply(rgb, **params)
#         return np.concatenate([rgb_aug, extra], axis=2)

# class NormalizeMultiChannel:
#     """
#     Placeholder for your existing multi-channel normalization.
#     Must implement __call__(image=...) -> dict(image=...)
#     """
#     def __init__(self, n_extra_channels: int):
#         self.n_extra_channels = n_extra_channels

#     def __call__(self, image, **kwargs):
#         # Example: ImageNet norm for RGB, leave extra channels as-is (or normalize them too)
#         rgb = image[:, :, :3].astype(np.float32) / 255.0
#         rgb = (rgb - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)

#         if self.n_extra_channels > 0:
#             extra = image[:, :, 3:].astype(np.float32)
#             # TODO: normalize extra channels if needed
#             image = np.concatenate([rgb, extra], axis=2)
#         else:
#             image = rgb

#         return {"image": image}

# def get_train_transforms(
#     img_size: int = 768,
#     p_hflip: float = 0.5,
#     p_vflip: float = 0.5,
#     p_rotate90: float = 0.5,
#     use_color_jitter: bool = True,
#     n_extra_channels: int = 0,

#     # --- Added augmentations (from the slides' recipe families) ---
#     p_jpeg: float = 0.3,            # JPEG artifacts / compression
#     jpeg_q_low: int = 30,
#     jpeg_q_high: int = 100,

#     p_blur_noise: float = 0.25,     # blur/noise group
#     p_dropout: float = 0.25,        # Random Erasing / Cutout style
#     max_holes: int = 8,
#     hole_size_frac: float = 0.12,   # hole size ~ img_size * this
# ) -> A.Compose:
#     """
#     Training augments:
#     - Geometric: flips + rotate90
#     - JPEG artifacts
#     - Blur/Noise
#     - Random Erasing / Cutout via CoarseDropout
#     - ColorJitter (RGB only)
#     - Normalize + ToTensor
#     """

#     transforms_list = []

#     # Resize
#     transforms_list.append(
#         A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA) if img_size is not None else A.NoOp()
#     )

#     # Geometric (works on all channels)
#     transforms_list += [
#         A.HorizontalFlip(p=p_hflip),
#         A.VerticalFlip(p=p_vflip),
#         A.RandomRotate90(p=p_rotate90),
#     ]

#     # JPEG artifacts (RGB-only if you have extra channels)
#     jpeg = A.ImageCompression(quality_lower=jpeg_q_low, quality_upper=jpeg_q_high, p=1.0)
#     if n_extra_channels > 0:
#         transforms_list.append(RGBOnlyTransform(jpeg, n_extra_channels=n_extra_channels, p=p_jpeg))
#     else:
#         transforms_list.append(A.ImageCompression(quality_lower=jpeg_q_low, quality_upper=jpeg_q_high, p=p_jpeg))

#     # Blur/Noise family (RGB-only is usually fine; for multi-channel, apply to RGB only)
#     blur_noise_block = A.OneOf(
#         [
#             A.GaussNoise(var_limit=(5.0, 30.0), p=1.0),
#             A.GaussianBlur(blur_limit=(3, 7), p=1.0),
#             A.MotionBlur(blur_limit=7, p=1.0),
#         ],
#         p=1.0,
#     )
#     if n_extra_channels > 0:
#         transforms_list.append(RGBOnlyTransform(blur_noise_block, n_extra_channels=n_extra_channels, p=p_blur_noise))
#     else:
#         transforms_list.append(blur_noise_block if p_blur_noise > 0 else A.NoOp())

#     # Cutout / Random Erasing: use CoarseDropout
#     # (Cutout is basically 1~few holes; Random Erasing is more general. CoarseDropout covers both.)
#     max_h = int(img_size * hole_size_frac)
#     max_w = int(img_size * hole_size_frac)
#     transforms_list.append(
#         A.CoarseDropout(
#             max_holes=max_holes,
#             max_height=max_h,
#             max_width=max_w,
#             min_holes=1,
#             min_height=max(1, max_h // 4),
#             min_width=max(1, max_w // 4),
#             fill_value=0,
#             p=p_dropout,
#         )
#     )

#     # ColorJitter only for pure RGB (your existing rule)
#     if use_color_jitter and n_extra_channels == 0:
#         transforms_list.append(
#             A.ColorJitter(
#                 brightness=0.2,
#                 contrast=0.2,
#                 saturation=0.2,
#                 hue=0.1,
#                 p=0.5,
#             )
#         )

#     # Normalize + ToTensor
#     if n_extra_channels > 0:
#         transforms_list.append(NormalizeMultiChannel(n_extra_channels=n_extra_channels))
#     else:
#         transforms_list.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

#     transforms_list.append(ToTensorV2())
#     return A.Compose(transforms_list)
    
# # ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# def get_train_transforms(
#     img_size: Optional[int] = 768,
#     p_hflip: float = 0.5,
#     p_vflip: float = 0.5,
#     p_rotate90: float = 0.5,
#     use_color_jitter: bool = True,
# ) -> A.Compose:
#     """
#     Get training augmentation transforms.
    
#     Strategy:
#     - HorizontalFlip, VerticalFlip, RandomRotate90 for geometric variety
#     - ColorJitter for color augmentation
#     - Applied independently to each patch
    
#     Args:
#         img_size: Target image size after resize (None = skip resize)
#         p_hflip: Probability of horizontal flip
#         p_vflip: Probability of vertical flip
#         p_rotate90: Probability of 90-degree rotation
#         use_color_jitter: Whether to apply color jitter
        
#     Returns:
#         Albumentations Compose object
#     """
#     transforms_list = []
    
#     # Resize to target size (skip if img_size is None)
#     if img_size is not None:
#         transforms_list.append(
#             A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA)
#         )
    
#     # Geometric augmentations
#     transforms_list.extend([
#         A.HorizontalFlip(p=p_hflip),
#         A.VerticalFlip(p=p_vflip),
#         A.RandomRotate90(p=p_rotate90),
#     ])
    
#     if use_color_jitter:
#         transforms_list.append(
#             A.ColorJitter(
#                 brightness=0.2,
#                 contrast=0.2,
#                 saturation=0.2,
#                 hue=0.1,
#                 p=0.5,
#             )
#         )
    
#     # Normalization and tensor conversion
#     transforms_list.extend([
#         A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
#         ToTensorV2(),
#     ])
    
#     return A.Compose(transforms_list)


# def get_train_transforms(img_size=768):
#     return A.Compose([
#         A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),

#         # Geometry: keep strong (helps generalization, doesn't corrupt color cues)
#         A.OneOf([
#             A.HorizontalFlip(p=1.0),
#             A.VerticalFlip(p=1.0),
#             A.RandomRotate90(p=1.0),
#         ], p=0.7),

#         A.ShiftScaleRotate(
#             shift_limit=0.05,
#             scale_limit=0.12,
#             rotate_limit=15,
#             border_mode=cv2.BORDER_REFLECT_101,
#             p=0.9
#         ),
#         A.Affine(shear={"x": (-4, 4), "y": (-4, 4)}, p=0.10),

#         # Photometric: keep, but "color-safe"
#         A.RandomBrightnessContrast(
#             brightness_limit=0.18,
#             contrast_limit=0.18,
#             p=0.65
#         ),
#         A.RandomGamma(gamma_limit=(90, 110), p=0.25),

#         # Reduce color shifts (important!)
#         A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.15),
#         A.HueSaturationValue(
#             hue_shift_limit=4,
#             sat_shift_limit=10,
#             val_shift_limit=10,
#             p=0.15
#         ),

#         # Field lighting
#         A.RandomShadow(p=0.25),

#         # Sensor artifacts: reduce frequency to preserve texture
#         A.OneOf([
#             A.ImageCompression(quality_lower=70, quality_upper=95, p=1.0),
#             A.GaussNoise(var_limit=(3.0, 15.0), p=1.0),
#             A.ISONoise(color_shift=(0.01, 0.04), intensity=(0.04, 0.20), p=1.0),
#         ], p=0.20),

#         A.OneOf([
#             A.MotionBlur(blur_limit=3, p=1.0),
#             A.GaussianBlur(blur_limit=3, p=1.0),
#         ], p=0.08),

#         # Occlusion: keep but mild
#         A.CoarseDropout(
#             max_holes=6,
#             max_height=int(img_size * 0.05),
#             max_width=int(img_size * 0.05),
#             min_holes=1,
#             fill_value=0,
#             p=0.15
#         ),

#         A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
#         ToTensorV2(),
#     ])

def get_train_transforms(img_size=768):
    return A.Compose([
        # --- Geometry ---
        A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),

        A.OneOf([
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.RandomRotate90(p=1.0),
        ], p=0.7),

        A.ShiftScaleRotate(
            shift_limit=0.05,     # up to 5% shift
            scale_limit=0.12,     # ±12% zoom
            rotate_limit=15,      # ±15°
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.9
        ),

        # Optional mild shear/affine (keep gentle)
        A.Affine(shear={"x": (-5, 5), "y": (-5, 5)}, p=0.15),

        # --- Photometric (vegetation-safe) ---
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.6
        ),
        A.RandomGamma(gamma_limit=(85, 115), p=0.3),

        # Channel shift (Narayanan used big shifts; keep moderate unless you verified you need more)
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.2),

        # Very mild hue/sat/value: avoid making green look dead
        A.HueSaturationValue(
            hue_shift_limit=6,
            sat_shift_limit=12,
            val_shift_limit=10,
            p=0.2
        ),

        # --- Sensor artifacts ---
        A.OneOf([
            A.ImageCompression(quality_lower=60, quality_upper=95, p=1.0),
            A.GaussNoise(var_limit=(5.0, 25.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.05, 0.25), p=1.0),
        ], p=0.35),

        A.OneOf([
            A.MotionBlur(blur_limit=3, p=1.0),
            A.GaussianBlur(blur_limit=3, p=1.0),
        ], p=0.2),

        # --- Field occlusions / shadows ---
        A.RandomShadow(p=0.2),

        A.CoarseDropout(
            max_holes=8,
            max_height=int(img_size * 0.06),
            max_width=int(img_size * 0.06),
            min_holes=1,
            fill_value=0,
            p=0.2
        ),

        # --- Normalize ---
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def get_val_transforms(img_size: Optional[int] = 768) -> A.Compose:
    """
    Get validation/test transforms (no augmentation).
    
    Args:
        img_size: Target image size after resize (None = skip resize)
        
    Returns:
        Albumentations Compose object
    """
    transforms_list = []
    
    if img_size is not None:
        transforms_list.append(
            A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA)
        )
    
    transforms_list.extend([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    
    return A.Compose(transforms_list)


def get_tta_transforms(img_size: Optional[int] = 768) -> List[A.Compose]:
    """
    TTA tuned for green/clover targets:
    - strong geometry (flips + small rotations)
    - very mild photometric (optional, but helps lighting robustness)
    - avoids hue/RGB shifts which can hurt green/clover separation
    """
    base = [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]

    resize_list = [A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA)] if img_size is not None else []

    def comp(aug_list):
        return A.Compose([*aug_list, *resize_list, *base])

    # Geometry (safe)
    original = comp([])
    hflip    = comp([A.HorizontalFlip(p=1.0)])
    vflip    = comp([A.VerticalFlip(p=1.0)])
    hvflip   = comp([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)])  # 180°

    rot_p5   = comp([A.Rotate(limit=(5, 5), border_mode=cv2.BORDER_REFLECT_101, p=1.0)])
    rot_m5   = comp([A.Rotate(limit=(-5, -5), border_mode=cv2.BORDER_REFLECT_101, p=1.0)])

    # Mild photometric (keep conservative for green/clover)
    # Only include these if you trained with brightness/contrast/gamma augments.
    # bc = comp([A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=1.0)])
    # gm = comp([A.RandomGamma(gamma_limit=(92, 108), p=1.0)])

    return [original, hflip, vflip, hvflip, rot_p5, rot_m5]


# def get_tta_transforms(img_size: Optional[int] = 768) -> list:
#     """
#     Get Test-Time Augmentation transforms.
    
#     Returns list of transforms for TTA:
#     1. Original (no augmentation)
#     2. Horizontal flip
#     3. Vertical flip
    
#     Args:
#         img_size: Target image size after resize (None = skip resize)
        
#     Returns:
#         List of Albumentations Compose objects
#     """
#     base = [
#         A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
#         ToTensorV2(),
#     ]
    
#     resize_list = [A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA)] if img_size is not None else []
    
#     original = A.Compose([
#         *resize_list,
#         *base,
#     ])
    
#     hflip = A.Compose([
#         A.HorizontalFlip(p=1.0),
#         *resize_list,
#         *base,
#     ])
    
#     vflip = A.Compose([
#         A.VerticalFlip(p=1.0),
#         *resize_list,
#         *base,
#     ])
    
#     return [original, hflip, vflip]


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
