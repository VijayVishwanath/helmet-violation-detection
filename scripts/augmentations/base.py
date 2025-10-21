import shutil
import os
from PIL import Image

class AugmentorBase:
    def __init__(self, image_folder, label_folder, output_image_folder, output_label_folder, file_suffix=None):
        """
        Base class initialization setting up folder paths and creating output dirs.
        
        :param image_folder: Directory containing original images.
        :param label_folder: Directory containing original label text files.
        :param output_image_folder: Folder path to save augmented images
        :param output_label_folder: Folder path to save augmented label files
        :param file_suffix: Suffix to append to output filenames (before extension)
        """
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.output_image_folder = output_image_folder
        self.output_label_folder = output_label_folder
        self.file_suffix = file_suffix  # e.g., 'rotation', 'flip', etc.

        os.makedirs(self.output_image_folder, exist_ok=True)
        os.makedirs(self.output_label_folder, exist_ok=True)

    def clear_output_folders(self):
        """
        Delete all files in the output image and label folders.
        """
        for folder in [self.output_image_folder, self.output_label_folder]:
            for fname in os.listdir(folder):
                fpath = os.path.join(folder, fname)
                if os.path.isfile(fpath):
                    os.remove(fpath)
                    
        print('clear completed')

    def get_augmented_filename(self, filename):
        """
        Returns filename with suffix inserted before extension if file_suffix is set.
        """
        if self.file_suffix:
            base, ext = os.path.splitext(filename)
            return f"{base}_{self.file_suffix}{ext}"
        return filename

    def process(self):
        """
        Each child class must implement its own process method.
        """
        # Clear output folders before processing
        raise NotImplementedError("Each augmentation class must implement its own process() method.")