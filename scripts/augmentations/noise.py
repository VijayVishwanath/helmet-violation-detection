
import os
import cv2
import numpy as np
from augmentations.base import AugmentorBase
from utils import copy_labels


class NoiseInjectionAugmentor(AugmentorBase):
    def __init__(self, image_folder, label_folder, output_image_folder, output_label_folder, noise_type='gaussian'):
        """
        Initialize NoiseInjectionAugmentor with source and destination folders and noise type.
        
        :param image_folder: Source folder with original images
        :param label_folder: Source folder with original labels
        :param output_image_folder: Destination folder to save noisy images
        :param output_label_folder: Destination folder to copy label files unchanged
        :param noise_type: 'gaussian' or 'salt_pepper'
        """
        super().__init__(image_folder, label_folder, output_image_folder, output_label_folder, file_suffix=noise_type)
        self.noise_type = noise_type
        
    
    def _add_gaussian_noise(self, img, mean=0, sigma=25):
        gauss = np.random.normal(mean, sigma, img.shape).astype(np.float32)
        noisy_img = cv2.add(img.astype(np.float32), gauss)
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        return noisy_img
    
    def _add_salt_pepper_noise(self, img, salt_prob=0.01, pepper_prob=0.01):
        noisy_img = img.copy()
        total_pixels = img.size // img.shape[2]
        # Salt noise
        num_salt = int(total_pixels * salt_prob)
        coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape[:2]]
        noisy_img[coords[0], coords[1], :] = 255
        # Pepper noise
        num_pepper = int(total_pixels * pepper_prob)
        coords = [np.random.randint(0, i -1, num_pepper) for i in img.shape[:2]]
        noisy_img[coords[0], coords[1], :] = 0
        return noisy_img
    
    def process(self):
        """
        Apply chosen noise injection to all images in source folder,
        save output and copy label files unchanged.
        """
        for fname in os.listdir(self.image_folder):
            if not fname.lower().endswith(('.jpg', '.png')):
                continue

            img_path = os.path.join(self.image_folder, fname)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
            
            if self.noise_type == 'gaussian':
                noisy_img = self._add_gaussian_noise(img)
            elif self.noise_type == 'salt_pepper':
                noisy_img = self._add_salt_pepper_noise(img)
            else:
                noisy_img = img.copy()

            dst_img_path = os.path.join(self.output_image_folder, self.get_augmented_filename(fname))
            cv2.imwrite(dst_img_path, noisy_img)
            
        # Use utility function to copy all label files at once
        copy_labels(self.label_folder, self.output_label_folder, self.file_suffix)
