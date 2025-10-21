import os
import random
import numpy as np
from PIL import Image
from augmentations.base import AugmentorBase
from utils import load_yolo_boxes

class MosaicAugmentor(AugmentorBase):
    
    def __init__(self, image_folder, label_folder, output_image_folder, output_label_folder, in_size=640):
        """
        Initialize the MosaicAugmentor with source and target folder paths and input size.

        :param image_folder: Folder path containing original images
        :param label_folder: Folder path containing original label files (YOLO format)
        :param output_image_folder: Folder path to save mosaic images
        :param output_label_folder: Folder path to save mosaic label files
        :param in_size: Size of final mosaic image (square)
        """
        super().__init__(image_folder, label_folder, output_image_folder, output_label_folder, file_suffix="mosaic")
        self.in_size = in_size
    
    def _mosaic_augment(self, image_files, label_files, output_image_path, output_label_path):
        cut_x = self.in_size//2 #random.randint(int(self.in_size*0.25), int(self.in_size*0.25))
        cut_y = self.in_size//2 #random.randint(int(self.in_size*0.25), int(self.in_size*0.25))
        
        mosaic_img = Image.new('RGB', (self.in_size, self.in_size), (114, 114, 114))
        mosaic_boxes = [] 
        
        for i, (img_path, label_path) in enumerate(zip(image_files, label_files)):
            img = Image.open(img_path).convert('RGB')
            w, h = img.size
            boxes = load_yolo_boxes(label_path, w, h)
            
            img = img.resize((self.in_size//2, self.in_size//2))
            scale_x = (self.in_size//2) / w
            scale_y = (self.in_size//2) / h
            
            adjusted_boxes = []
            for x1, y1, x2, y2, cls in boxes:
                adjusted_boxes.append([x1*scale_x, y1*scale_y, x2*scale_x, y2*scale_y, cls])
            
            if i == 0:
                paste_pos = (0, 0)
                x_offset, y_offset = 0, 0
            elif i == 1:
                paste_pos = (cut_x, 0)
                x_offset, y_offset = cut_x, 0
            elif i == 2:
                paste_pos = (0, cut_y)
                x_offset, y_offset = 0, cut_y
            else:
                paste_pos = (cut_x, cut_y)
                x_offset, y_offset = cut_x, cut_y
            
            mosaic_img.paste(img, paste_pos)
            
            for x1, y1, x2, y2, cls in adjusted_boxes:
                nx1 = np.clip(x1 + x_offset, 0, self.in_size)
                ny1 = np.clip(y1 + y_offset, 0, self.in_size)
                nx2 = np.clip(x2 + x_offset, 0, self.in_size)
                ny2 = np.clip(y2 + y_offset, 0, self.in_size)
                
                if nx2 - nx1 > 1 and ny2 - ny1 > 1:
                    mosaic_boxes.append([nx1, ny1, nx2, ny2, cls])
        
        with open(output_label_path, 'w') as f:
            for bx1, by1, bx2, by2, cls in mosaic_boxes:
                cx = (bx1 + bx2)/2 / self.in_size
                cy = (by1 + by2)/2 / self.in_size
                w = (bx2 - bx1) / self.in_size
                h = (by2 - by1) / self.in_size
                f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        
        mosaic_img.save(output_image_path)
    
    def _get_image_label_paths(self):
        img_files = [f for f in os.listdir(self.image_folder) if f.lower().endswith(('.jpg', '.png'))]
        img_paths = [os.path.join(self.image_folder, f) for f in img_files]
        label_paths = [os.path.join(self.label_folder, os.path.splitext(f)[0] + '.txt') for f in img_files]

        valid_pairs = [(img, lbl) for img, lbl in zip(img_paths, label_paths) if os.path.exists(lbl)]
        return valid_pairs
    
    def process(self):
        valid_pairs = self._get_image_label_paths()
        random.shuffle(valid_pairs)
        
        for i in range(1, len(valid_pairs)+1, 4):
            batch = valid_pairs[i-1:i+3]
            if len(batch) < 4:
                print(f"Skipping last batch starting at index {i}, not enough images for mosaic")
                break
            
            image_files = [b[0] for b in batch]
            label_files = [b[1] for b in batch]
            
            mosaic_filename = f'mosaic_{i//4}.jpg'
            mosaic_labelname = f'mosaic_{i//4}.txt'
            mosaic_img_path = os.path.join(self.output_image_folder, mosaic_filename)
            mosaic_lbl_path = os.path.join(self.output_label_folder, mosaic_labelname)

            self._mosaic_augment(image_files, label_files, mosaic_img_path, mosaic_lbl_path)
