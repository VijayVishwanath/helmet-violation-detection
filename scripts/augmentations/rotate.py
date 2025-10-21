import os
import shutil
import cv2
import albumentations as A
from augmentations.base import AugmentorBase
from utils import copy_labels


class RotateAugmentor(AugmentorBase):
    def __init__(self, image_folder, label_folder, output_image_folder, output_label_folder, max_angle=20):
        """
        Initialize AlbumentationsRotateAugmentor with source and destination folders,
        and the maximum rotation angle.
        
        :param input_image_folder: Source folder containing images
        :param input_label_folder: Source folder containing labels (YOLO format)
        :param output_image_folder: Destination folder to save rotated images
        :param output_label_folder: Destination folder to save rotated labels
        :param max_angle: Maximum absolute rotation angle (degrees)
        """
        super().__init__(image_folder, label_folder, output_image_folder, output_label_folder, file_suffix="rotation")
        self.max_angle = max_angle        
    
       
    def process(self):
        self.clear_output_folders()
        # Copy labels first (optional, overwrite after transform)
        copy_labels(self.label_folder, self.output_label_folder, self.file_suffix)
        
        transform = A.Compose([
            A.Rotate(limit=self.max_angle, border_mode=cv2.BORDER_CONSTANT, value=(114,114,114), p=1.0)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        image_files = [f for f in os.listdir(self.image_folder) if f.lower().endswith(('.jpg', '.png'))]
        
        for fname in image_files:
            img_path = os.path.join(self.image_folder, fname)
            label_path = os.path.join(self.label_folder, os.path.splitext(fname)[0] + '.txt')
            
            img = cv2.imread(img_path)
            
            if img is None:
                continue
            
            h, w = img.shape[:2]
            
            bboxes = []
            class_labels = []
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        cls, cx, cy, bw, bh = line.strip().split()
                        bboxes.append([float(cx), float(cy), float(bw), float(bh)])
                        class_labels.append(int(cls))
            
            if bboxes:
                augmented = transform(image=img, bboxes=bboxes, class_labels=class_labels)
                aug_img = augmented['image']
                aug_bboxes = augmented['bboxes']
                aug_labels = augmented['class_labels']

                dst_img_path = os.path.join(self.output_image_folder, self.get_augmented_filename(fname))
                cv2.imwrite(dst_img_path, aug_img)

                label_fname = os.path.basename(label_path)
                dst_label_path = os.path.join(self.output_label_folder, self.get_augmented_filename(label_fname))
                if os.path.exists(dst_label_path):
                    os.chmod(dst_label_path, 0o777)  # Make writable (on Unix)
                    os.remove(dst_label_path)
                
                with open(dst_label_path, 'w') as f:
                    if aug_bboxes:
                        for bbox, label in zip(aug_bboxes, aug_labels):
                            cx, cy, w_n, h_n = bbox
                            f.write(f"{int(label)} {cx:.6f} {cy:.6f} {w_n:.6f} {h_n:.6f}\n")
                    else:
                        f.write("")
            else:
                # No bboxes: copy original image and label
                cv2.imwrite(os.path.join(self.output_image_folder, self.get_augmented_filename(fname)), img)
                if os.path.exists(label_path):
                    shutil.copy2(label_path, self.output_label_folder)
