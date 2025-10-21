import os
from PIL import Image
from augmentations.base import AugmentorBase

class HorizontalFlip(AugmentorBase):
    
    def __init__(self, image_folder, label_folder, output_image_folder, output_label_folder):
        super().__init__(image_folder, label_folder, output_image_folder, output_label_folder, file_suffix="flip")
        
    def _flip_images(self):
        """Flip images horizontally and save with '_flip' suffix."""
        for fname in os.listdir(self.image_folder):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(self.image_folder, fname)
                img = Image.open(img_path)
                mirrored_img = img.transpose(Image.FLIP_LEFT_RIGHT)

                mirrored_img.save(os.path.join(self.output_image_folder, self.get_augmented_filename(fname)))
    
    def _flip_labels(self):
        """Flip YOLO label x_center coordinates and save with '_flip' suffix."""
        for fname in os.listdir(self.label_folder):
            if fname.lower().endswith('.txt'):
                mirrored_fname = self.get_augmented_filename(fname)
                with open(os.path.join(self.label_folder, fname), 'r') as f_in, \
                    open(os.path.join(self.output_label_folder, mirrored_fname), 'w') as f_out:
                    for line in f_in:
                        parts = line.strip().split()
                        class_id = parts[0]
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])

                        mirrored_x_center = 1.0 - x_center
                        new_line = f"{class_id} {mirrored_x_center} {y_center} {width} {height}\n"
                        f_out.write(new_line)
    
    def process(self):
        self.clear_output_folders()
        """Flip both images and labels."""
        self._flip_images()
        self._flip_labels()