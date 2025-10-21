# utils.py
import os
import shutil
import random
import cv2
from matplotlib import pyplot as plt

def load_yolo_boxes(label_path, img_width, img_height):
    """
    Load boxes from YOLO label file and convert normalized coordinates to absolute coords.
    
    Returns list of [x1, y1, x2, y2, class_id].
    """
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            cx = float(parts[1]) * img_width
            cy = float(parts[2]) * img_height
            w = float(parts[3]) * img_width
            h = float(parts[4]) * img_height
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            boxes.append([x1, y1, x2, y2, class_id])
    return boxes



def copy_labels(source_label_folder, target_label_folder, suffix=None):
    """
    Copy all .txt label files from source folder to target folder.
    Optionally add a suffix before the file extension.

    :param source_label_folder: Path to folder containing original label files
    :param target_label_folder: Path to folder to copy label files into
    :param suffix: Optional string to append before '.txt' in the filename
    """
    os.makedirs(target_label_folder, exist_ok=True)
    
    for fname in os.listdir(source_label_folder):
        if fname.endswith('.txt'):
            src = os.path.join(source_label_folder, fname)
            if suffix:
                base, ext = os.path.splitext(fname)
                dst_fname = f"{base}_{suffix}{ext}"
            else:
                dst_fname = fname
            dst = os.path.join(target_label_folder, dst_fname)
            shutil.copy2(src, dst)
            
def save_yolo_boxes(label_path, boxes, img_w, img_h):
    with open(label_path, 'w') as f:
        for x1, y1, x2, y2, cls in boxes:
            cx = (x1 + x2) / 2 / img_w
            cy = (y1 + y2) / 2 / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            
def draw_yolo_bboxes(image_folder, label_folder, filename, class_map=None):
    """
    Draw YOLO bounding boxes on a specific image and return the annotated image (RGB, np.array).
    """
    # Build paths
    image_path = os.path.join(image_folder, filename)
    label_path = os.path.join(label_folder, os.path.splitext(filename)[0] + ".txt")
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    colors = {0:(255,0,0),1:(0,255,0),2:(0,255,255),3:(255,165,0),4:(0,0,255)}
    # Draw bounding boxes
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for label in f.readlines():
                parts = label.strip().split()
                class_id = int(parts[0])
                x_center, y_center, bw, bh = map(float, parts[1:])
                x_center, y_center, bw, bh = x_center * w, y_center * h, bw * w, bh * h
                x1, y1 = int(x_center - bw / 2), int(y_center - bh / 2)
                x2, y2 = int(x_center + bw / 2), int(y_center + bh / 2)
                cv2.rectangle(image, (x1, y1), (x2, y2), colors.get(class_id, (255,255,255)), 2)
                label_text = class_map[class_id] if class_map and class_id in class_map else str(class_id)
                cv2.putText(image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, colors.get(class_id, (255,255,255)), 2)
    return image
            
def show_images_grid(image_folders, label_folders, filenames, class_map=None):
    """
    Display a grid of annotated images, given lists of image_folders, label_folders, and filenames.
    Each entry at position i in the arrays should correspond to the same image-label pair.
    """
    N = len(filenames)
    cols = 2
    rows = (N + 1) // cols
    plt.figure(figsize=(15, 7 * rows))
    for idx in range(N):
        image_folder=image_folders[idx],
        label_folder=label_folders[idx],
        filename=filenames[idx],
        img_annotated = draw_yolo_bboxes(
            image_folder=image_folder[0],
            label_folder=label_folder[0],
            filename=filename[0],
            class_map=class_map
        )
        plt.subplot(rows, cols, idx+1)
        plt.imshow(img_annotated)
        plt.axis("off")
        plt.title(filenames[idx])
    plt.tight_layout()
    plt.show()

def split_and_copy_dataset(
    src_img_dir, src_lbl_dir,
    dst_train_img_dir, dst_train_lbl_dir,
    dst_val_img_dir, dst_val_lbl_dir,
    split_ratio=0.8,
    seed=42
):
    """
    Copies images and labels from raw/train to model/train and model/val in the given split ratio.
    Clears destination folders before copying. Never modifies the raw dataset.
    """
    # Ensure destination folders exist and are empty
    for d in [dst_train_img_dir, dst_train_lbl_dir, dst_val_img_dir, dst_val_lbl_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    # List all images (assuming .jpg, change if needed)
    images = [f for f in os.listdir(src_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.seed(seed)
    random.shuffle(images)

    split_idx = int(len(images) * split_ratio)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    def copy_files(img_list, img_dst, lbl_dst):
        for img_name in img_list:
            base = os.path.splitext(img_name)[0]
            lbl_name = base + ".txt"
            src_img_path = os.path.join(src_img_dir, img_name)
            src_lbl_path = os.path.join(src_lbl_dir, lbl_name)
            dst_img_path = os.path.join(img_dst, img_name)
            dst_lbl_path = os.path.join(lbl_dst, lbl_name)
            if os.path.exists(src_img_path) and os.path.exists(src_lbl_path):
                shutil.copy2(src_img_path, dst_img_path)
                shutil.copy2(src_lbl_path, dst_lbl_path)

    copy_files(train_imgs, dst_train_img_dir, dst_train_lbl_dir)
    copy_files(val_imgs, dst_val_img_dir, dst_val_lbl_dir)
    #print(f"Copied {len(train_imgs)} images to train and {len(val_imgs)} images to val.")


def show_random_images_grid(image_folder, label_folder, class_map=None, N=4):
    filenames = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]
    random_files = random.sample(filenames, min(N, len(filenames)))
    cols = 2
    rows = (len(random_files) + 1) // cols
    plt.figure(figsize=(15, 7 * rows))
    for idx, filename in enumerate(random_files):
        img_annotated = draw_yolo_bboxes(
            image_folder=image_folder,
            label_folder=label_folder,
            filename=filename,
            class_map=class_map
        )
        plt.subplot(rows, cols, idx+1)
        plt.imshow(img_annotated)
        plt.axis("off")
        plt.title(filename)
    plt.tight_layout()
    plt.show()
