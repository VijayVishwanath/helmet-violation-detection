import os
import cv2
import numpy as np
from augmentations.base import AugmentorBase
from utils import copy_labels

class ShadowCastingAugmentor(AugmentorBase):
    
    def __init__(self, image_folder, label_folder, output_image_folder, output_label_folder):
        super().__init__(image_folder, label_folder, output_image_folder, output_label_folder, file_suffix="shadow")
        
    def _add_shadow(self, img, shadow_roi=None):
        """
        Add a polygon shadow overlay on the input image.
        
        :param img: Input image as numpy array
        :param shadow_roi: Optional polygon vertices [(x1,y1), (x2,y2),...] for shadow region
        :return: Shadowed image as numpy array
        """
        h, w = img.shape[:2]
        if shadow_roi is None:
            top_x = w * np.random.uniform(0, 1)
            top_y = 0
            bot_x = w * np.random.uniform(0, 1)
            bot_y = h
            vertices = np.array([[(top_x, top_y), (bot_x, bot_y), (w, bot_y), (w, top_y)]], dtype=np.int32)
        else:
            vertices = np.array([shadow_roi], dtype=np.int32)

        overlay = img.copy()
        cv2.fillPoly(overlay, vertices, (0, 0, 0))
        alpha = np.random.uniform(0.2, 0.4)  # Shadow intensity
        shadowed = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        return shadowed

    def process(self):
        """
        Apply shadow augmentation to all images in source folder, 
        saving results and copying labels unchanged.
        """
        for fname in os.listdir(self.image_folder):
            if not fname.lower().endswith(('.jpg', '.png')):
                continue

            img_path = os.path.join(self.image_folder, fname)

            img = cv2.imread(img_path)
            img_shadow = self._add_shadow(img)

            dst_img_path = os.path.join(self.output_image_folder, self.get_augmented_filename(fname))
            cv2.imwrite(dst_img_path, img_shadow)

        # Use utility function to copy all label files at once
        copy_labels(self.label_folder, self.output_label_folder, self.file_suffix)
