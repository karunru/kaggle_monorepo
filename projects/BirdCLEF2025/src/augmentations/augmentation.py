import torch
import torchvision.transforms.v2 as T
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
import numpy as np


IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB
IMAGENET_STD = [0.229, 0.224, 0.225]  # RGB


def pet_transform():
    transform = {
        "train": T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        ),
        "val": T.Compose(
            [
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        ),
    }
    return transform


def inaugment_correct_paste(
    image, n_patches=5, patch_size_ratio=0.1, resize_factors=(0.5, 1.5)
):
    """
    Apply InAugment with correct pasting behavior: Copy patches, resize and paste them back into the image.

    Parameters:
    - image: Input image as a NumPy array of shape (H, W, C).
    - n_patches: Number of patches to copy from the input image.
    - patch_size_ratio: Ratio of the patch size compared to the image size.
    - resize_factors: Tuple of factors (min, max) for resizing the patches.

    Returns:
    - Augmented image as a NumPy array of the same shape as the input.
    """
    H, W, C = image.shape
    augmented_image = image.copy()

    # Calculate patch size based on the image size and the given ratio
    patch_size = int(min(H, W) * patch_size_ratio)

    for _ in range(n_patches):
        # Randomly select the top-left corner of the patch
        y = np.random.randint(0, H - patch_size)
        x = np.random.randint(0, W - patch_size)

        # Extract the patch
        patch = image[y : y + patch_size, x : x + patch_size, :]

        # Randomly choose a resize factor and resize the patch
        resize_factor = np.random.uniform(*resize_factors)
        target_size = int(patch_size * resize_factor)
        resized_patch = transform.resize(
            patch, (target_size, target_size), anti_aliasing=True
        )

        # Convert resized patch back to original color range
        resized_patch = (resized_patch * 255).astype(image.dtype)

        # Choose a random location to paste the resized patch
        paste_y = np.random.randint(0, H - target_size)
        paste_x = np.random.randint(0, W - target_size)

        # Paste the resized patch directly into the image, without blending
        augmented_image[
            paste_y : paste_y + target_size, paste_x : paste_x + target_size
        ] = resized_patch

    return augmented_image


class InAugment(ImageOnlyTransform):
    def __init__(
        self,
        n_patches=3,
        patch_size_ratio=0.2,
        resize_factors=(0.5, 1.5),
        always_apply=False,
        p=0.5,
    ):
        super(InAugment, self).__init__(always_apply, p)
        self.n_patches = n_patches
        self.patch_size_ratio = patch_size_ratio
        self.resize_factors = resize_factors

    def apply(self, img, **params):
        return inaugment_correct_paste(
            img,
            n_patches=self.n_patches,
            patch_size_ratio=self.patch_size_ratio,
            resize_factors=self.resize_factors,
        )


def get_default_transforms():

    params1 = {
        "num_masks_x": 1,
        "mask_x_length": (0, 20),  # This line changed from fixed  to a range
        "fill_value": (0, 1, 2, 3, 4, 5, 6, 7),
    }
    params2 = {
        "num_masks_y": 1,
        "mask_y_length": (0, 20),
        "fill_value": (0, 1, 2, 3, 4, 5, 6, 7),
    }
    transform = {
        "albu_train": A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                InAugment(
                    n_patches=3,
                    patch_size_ratio=0.2,
                    resize_factors=(0.5, 1.5),
                    p=0.5,
                ),
                A.XYMasking(**params1, p=0.3),
                A.XYMasking(**params2, p=0.3),
            ]
        ),
        "train": T.Compose(
            [
                T.RandomVerticalFlip(p=0.0),
            ]
        ),
        "albu_val": A.Compose(
            [
                A.NoOp(always_apply=True),
            ]
        ),
        "val": T.Compose(
            [
                T.RandomHorizontalFlip(p=0.0),
            ]
        ),
    }
    return transform
