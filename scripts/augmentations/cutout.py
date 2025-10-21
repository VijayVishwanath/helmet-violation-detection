import os
import random
import numpy as np
from PIL import Image, ImageDraw
from augmentations.base import AugmentorBase
from utils import copy_labels


class CutoutAugmentor(AugmentorBase):
    def __init__(self, image_folder, label_folder, output_image_folder, output_label_folder,
                 n_holes=2, min_length=30, max_length=80):
        """
        Initialize CutoutAugmentor with folders and cutout parameters.

        :param image_folder: Folder containing original images
        :param label_folder: Folder containing original labels
        :param output_image_folder: Folder to save cutout augmented images
        :param output_label_folder: Folder to copy labels to
        :param n_holes: Number of cutout holes per image
        :param min_length: Minimum width/height of cutout hole
        :param max_length: Maximum width/height of cutout hole
        """
        super().__init__(image_folder, label_folder, output_image_folder, output_label_folder, file_suffix="cutout")
        self.n_holes = n_holes
        self.min_length = min_length
        self.max_length = max_length        
    
    def _cutout(self, img):
        """Apply CutOut augmentation on a single PIL image."""
        w, h = img.size
        draw = ImageDraw.Draw(img)
        for _ in range(self.n_holes):
            length = random.randint(self.min_length, self.max_length)
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            x1 = np.clip(x - length // 2, 0, w)
            y1 = np.clip(y - length // 2, 0, h)
            x2 = np.clip(x + length // 2, 0, w)
            y2 = np.clip(y + length // 2, 0, h)
            draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))
        return img
    
    def process(self):
        self.clear_output_folders()
        """Apply cutout augmentation to all images in source folder and save to target folder."""
        img_files = [f for f in os.listdir(self.image_folder) if f.lower().endswith(('.jpg', '.png'))]
        for fname in img_files:
            img_path = os.path.join(self.image_folder, fname)
            img = Image.open(img_path).convert('RGB')
            aug_img = self._cutout(img)
            aug_img.save(os.path.join(self.output_image_folder, self.get_augmented_filename(fname)))
        copy_labels(self.label_folder, self.output_label_folder, suffix=self.file_suffix)
