import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from augmentations.base import AugmentorBase
from utils import copy_labels


class SyntheticImageAugmentor(AugmentorBase):
    def __init__(self, image_folder, label_folder, output_image_folder, output_label_folder):
        """
        Initialize the SyntheticImageAugmentor with source and destination folders.
        
        :param image_folder: Folder containing original images
        :param label_folder: Folder containing original labels
        :param output_image_folder: Folder to save cutout augmented images
        :param output_label_folder: Folder to copy labels to
        """
        super().__init__(image_folder, label_folder, output_image_folder, output_label_folder)
        self.aug_types = ["blur", "fog", "rain", "illumination"]
    
    def _add_blur(self, img, max_radius=4):
        radius = random.uniform(1, max_radius)
        return img.filter(ImageFilter.GaussianBlur(radius))
    
    def _add_illumination(self, img, min_fac=0.6, max_fac=1.5):
        enhancer = ImageEnhance.Brightness(img)
        fac = random.uniform(min_fac, max_fac)
        return enhancer.enhance(fac)
    
    def _add_fog(self, np_img, fog_intensity=0.5):
        h, w = np_img.shape[:2]
        center_x, center_y = np.random.randint(0, w), np.random.randint(0, h)
        fog = np.zeros_like(np_img, dtype=np.float32)
        radius = int(min(h, w) * np.random.uniform(0.5, 1))
        cv2.circle(fog, (center_x, center_y), radius, (255, 255, 255), -1)
        fog = cv2.GaussianBlur(fog, (0, 0), sigmaX=radius//2)
        fog_alpha = np.clip(fog * fog_intensity / 255.0, 0, 1)
        result = np_img * (1 - fog_alpha) + fog * fog_alpha
        return result.astype(np.uint8)
    
    def _add_rain(self, np_img, rain_intensity=0.3, drop_length=20, drop_width=1):
        h, w = np_img.shape[:2]
        rain_img = np_img.copy()
        num_drops = int(rain_intensity * h * w / 100)
        for _ in range(num_drops):
            x1 = np.random.randint(0, w)
            y1 = np.random.randint(0, h)
            x2 = x1 + np.random.randint(-2, 2)
            y2 = y1 + drop_length
            cv2.line(rain_img, (x1, y1), (x2, y2), (200, 200, 200), drop_width)
        return cv2.blur(rain_img, (3, 3))
    
    def _augment_image_file(self, img_path, aug_type):
        img = Image.open(img_path).convert('RGB')
        if aug_type == "blur":
            return self._add_blur(img)
        elif aug_type == "illumination":
            return self._add_illumination(img)
        else:
            np_img = np.array(img)
            if aug_type == "fog":
                return Image.fromarray(self._add_fog(np_img))
            elif aug_type == "rain":
                return Image.fromarray(self._add_rain(np_img))
        return img
    
    def process(self):
        self.clear_output_folders()
        """
        Perform synthetic augmentation on all images in source directory,
        save augmented images and copy corresponding label files unchanged.
        """
        img_files = [f for f in os.listdir(self.image_folder) if f.lower().endswith(('.jpg', '.png'))]
        for fname in img_files:
            aug_type = random.choice(self.aug_types)
            img_path = os.path.join(self.image_folder, fname)
            base_name = os.path.splitext(fname)[0]
            target_img_path = os.path.join(self.output_image_folder, base_name + '_' + aug_type + '.jpg')

            aug_img = self._augment_image_file(img_path, aug_type)
            aug_img.save(target_img_path)
            
            # Copy label with matching suffix
            label_src = os.path.join(self.label_folder, base_name + '.txt')
            label_dst = os.path.join(self.output_label_folder, f"{base_name}_{aug_type}.txt")
            if os.path.exists(label_src):
                os.makedirs(self.output_label_folder, exist_ok=True)
                with open(label_src, 'r') as fin, open(label_dst, 'w') as fout:
                    fout.write(fin.read())
        
