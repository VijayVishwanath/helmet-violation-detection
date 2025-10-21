import os
import random
import numpy as np
from PIL import Image
from augmentations.base import AugmentorBase
from utils import load_yolo_boxes, save_yolo_boxes

class CutMixAugmentor(AugmentorBase):
    def __init__(self, image_folder, label_folder, output_image_folder, output_label_folder):
        super().__init__(image_folder, label_folder, output_image_folder, output_label_folder, file_suffix="cutmix")
        
    def _cutmix_two_images(self, img1, boxes1, img2, boxes2):
        w, h = img1.size
        cut_w = int(w * random.uniform(0.3, 0.7))
        cut_h = int(h * random.uniform(0.3, 0.7))
        cut_x = random.randint(0, w - cut_w)
        cut_y = random.randint(0, h - cut_h)

        img1_np = np.array(img1)
        img2_np = np.array(img2)

        img1_np[cut_y:cut_y+cut_h, cut_x:cut_x+cut_w] = img2_np[cut_y:cut_y+cut_h, cut_x:cut_x+cut_w]
        mixed_img = Image.fromarray(img1_np)

        filtered_boxes1 = []
        for x1, y1, x2, y2, cls in boxes1:
            inter_x1 = max(x1, cut_x)
            inter_y1 = max(y1, cut_y)
            inter_x2 = min(x2, cut_x + cut_w)
            inter_y2 = min(y2, cut_y + cut_h)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            box_area = (x2 - x1) * (y2 - y1)
            if box_area == 0:
                continue
            if inter_area / box_area <= 0.5:
                filtered_boxes1.append([x1, y1, x2, y2, cls])

        filtered_boxes2 = []
        for x1, y1, x2, y2, cls in boxes2:
            inter_x1 = max(x1, cut_x)
            inter_y1 = max(y1, cut_y)
            inter_x2 = min(x2, cut_x + cut_w)
            inter_y2 = min(y2, cut_y + cut_h)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            if inter_area <= 0:
                continue
            nx1 = max(x1, cut_x)
            ny1 = max(y1, cut_y)
            nx2 = min(x2, cut_x + cut_w)
            ny2 = min(y2, cut_y + cut_h)
            filtered_boxes2.append([nx1, ny1, nx2, ny2, cls])

        new_boxes = filtered_boxes1 + filtered_boxes2
        return mixed_img, new_boxes

    def process(self):
        self.clear_output_folders()
        img_files = [f for f in os.listdir(self.image_folder) if f.lower().endswith(('.jpg', '.png'))]
        random.shuffle(img_files)

        for i in range(0, len(img_files) - 1, 2):
            img1_path = os.path.join(self.image_folder, img_files[i])
            img2_path = os.path.join(self.image_folder, img_files[i+1])
            label1_path = os.path.join(self.label_folder, os.path.splitext(img_files[i])[0] + '.txt')
            label2_path = os.path.join(self.label_folder, os.path.splitext(img_files[i+1])[0] + '.txt')

            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
            w1, h1 = img1.size
            w2, h2 = img2.size

            boxes1 = load_yolo_boxes(label1_path, w1, h1)
            boxes2 = load_yolo_boxes(label2_path, w2, h2)

            mixed_img, new_boxes = self._cutmix_two_images(img1, boxes1, img2, boxes2)

            img1_fname = os.path.basename(img1_path)
            label1_fname = os.path.basename(label1_path)
            out_img_path = os.path.join(self.output_image_folder, self.get_augmented_filename(img1_fname))
            out_label_path = os.path.join(self.output_label_folder, self.get_augmented_filename(label1_fname))
            mixed_img.save(out_img_path)
            save_yolo_boxes(out_label_path, new_boxes, w1, h1)
