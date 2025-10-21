
import os
import cv2
from augmentations.base import AugmentorBase
from utils import copy_labels


class GrayscaleAugmentor(AugmentorBase):
    
    def __init__(self, image_folder, label_folder, output_image_folder, output_label_folder):
        super().__init__(image_folder, label_folder, output_image_folder, output_label_folder, file_suffix="grayscale")
        
    def _convert_grayscale(self, img):
        """
        Convert BGR color image to grayscale and back to 3-channel BGR.
        
        :param img: Input image as numpy array (BGR)
        :return: Grayscale 3-channel image as numpy array (BGR)
        """
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray_3ch = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        return img_gray_3ch

    def process(self):
        """
        Convert all images in source folder to grayscale images,
        save output and copy label files unchanged.
        """
        for fname in os.listdir(self.image_folder):
            if not fname.lower().endswith(('.jpg', '.png')):
                continue

            src_img_path = os.path.join(self.image_folder, fname)
            dst_img_path = os.path.join(self.output_image_folder, self.get_augmented_filename(fname))

            img_color = cv2.imread(src_img_path)
            img_gray_3ch = self._convert_grayscale(img_color)
            cv2.imwrite(dst_img_path, img_gray_3ch)

        # Use utility function to copy all label files at once
        copy_labels(self.label_folder, self.output_label_folder, self.file_suffix)
