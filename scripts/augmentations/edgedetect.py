import os
import cv2
from augmentations.base import AugmentorBase
from utils import copy_labels


class EdgeDetectAugmentor(AugmentorBase):
    
    def __init__(self, image_folder, label_folder, output_image_folder, output_label_folder):
        super().__init__(image_folder, label_folder, output_image_folder, output_label_folder, file_suffix="edge")
    
    def _edge_detect_image(self, input_path, output_path):
        """
        Apply Canny edge detection on a single image and save output color image.
        """
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1=100, threshold2=200)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(output_path, edges_colored)
    
    def process(self):
        self.clear_output_folders()
        """
        Process all images in source folder with edge detection and copy labels unchanged.
        """
        for fname in os.listdir(self.image_folder):
            if not (fname.lower().endswith('.jpg') or fname.lower().endswith('.png')):
                continue

            src_img_path = os.path.join(self.image_folder, fname)
            dst_img_path = os.path.join(self.output_image_folder, self.get_augmented_filename(fname))
            self._edge_detect_image(src_img_path, dst_img_path)

        # Use utility function to copy all label files at once
        copy_labels(self.label_folder, self.output_label_folder, self.file_suffix)
