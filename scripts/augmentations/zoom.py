import os
import random
from PIL import Image
from augmentations.base import AugmentorBase
from utils import load_yolo_boxes

class DynamicZoomer(AugmentorBase):

    def __init__(self, image_folder, label_folder, output_image_folder, output_label_folder, padding=30):
        """
        Initialize the DynamicZoomer with source and target folder paths and padding for cropping.

        :param image_folder: Folder path containing original images
        :param label_folder: Folder path containing original label files (YOLO format)
        :param output_image_folder: Folder path to save zoomed images
        :param output_label_folder: Folder path to save zoomed label files
        :param padding: Padding pixels added around the bounding box crop
        :param file_suffix: Suffix to append to output filenames (before extension)
        """
        super().__init__(image_folder, label_folder, output_image_folder, output_label_folder, file_suffix="zoom")
        self.padding = padding
        
    def _get_padded_crop_box(self, boxes, img_width, img_height):
        """
        Compute padded bounding box that contains all boxes with random padding.
        
        Returns (x1, y1, x2, y2).
        """
        padding = random.randint(self.padding, self.padding)
        x1 = max(0, min(b[0] for b in boxes) - padding)
        y1 = max(0, min(b[1] for b in boxes) - padding)
        x2 = min(img_width, max(b[2] for b in boxes) + padding)
        y2 = min(img_height, max(b[3] for b in boxes) + padding)
        return x1, y1, x2, y2
    
    def _dynamic_zoom(self, img_path, label_path):
        """
        Perform dynamic zoom: crop image tightly around boxes with padding, 
        resize back to original size, and adjust box coordinates.
        
        Returns resized PIL image and updated YOLO boxes.
        """
        img = Image.open(img_path)
        width, height = img.size
        boxes = load_yolo_boxes(label_path, width, height)
        
        crop_x1, crop_y1, crop_x2, crop_y2 = self._get_padded_crop_box(boxes, width, height)
        
        cropped_img = img.crop((int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)))
        resized_img = cropped_img.resize((width, height), Image.Resampling.LANCZOS)
        
        scale_x = width / (crop_x2 - crop_x1)
        scale_y = height / (crop_y2 - crop_y1)
        
        new_boxes = []
        for x1, y1, x2, y2, class_id in boxes:
            nx1 = max(0, x1 - crop_x1) * scale_x
            ny1 = max(0, y1 - crop_y1) * scale_y
            nx2 = min(crop_x2 - crop_x1, x2 - crop_x1) * scale_x
            ny2 = min(crop_y2 - crop_y1, y2 - crop_y1) * scale_y
            
            cx_n = (nx1 + nx2) / 2 / width
            cy_n = (ny1 + ny2) / 2 / height
            w_n = (nx2 - nx1) / width
            h_n = (ny2 - ny1) / height
            
            new_boxes.append([class_id, cx_n, cy_n, w_n, h_n])
        
        return resized_img, new_boxes
    
    def process(self):
        self.clear_output_folders()
        """
        Process dynamic zooming on all images and labels in source folders,
        save zoomed images and updated labels to target folders.
        """
        for filename in os.listdir(self.image_folder):
            if not filename.lower().endswith(('.jpg', '.png')):
                continue

            img_path = os.path.join(self.image_folder, filename)
            label_name = os.path.splitext(filename)[0] + '.txt'
            label_path = os.path.join(self.label_folder, label_name)

            if not os.path.exists(label_path):
                print(f"Label file missing for {filename}, skipping.")
                continue
            
            zoomed_img, zoomed_labels = self._dynamic_zoom(img_path, label_path)
            
            target_img_path = os.path.join(self.output_image_folder, self.get_augmented_filename(filename))
            target_label_path = os.path.join(self.output_label_folder, self.get_augmented_filename(label_name))

            zoomed_img.save(target_img_path)
            
            with open(target_label_path, 'w') as f:
                for b in zoomed_labels:
                    f.write(f"{int(b[0])} {b[1]:.8f} {b[2]:.8f} {b[3]:.8f} {b[4]:.8f}\n")
