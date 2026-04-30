import os
import glob
import random
import shutil

dataset_base = r"C:\Users\DHRUV CHAUHAN\Desktop\PROJECT\code\Weld_dataset"
train_img_dir = os.path.join(dataset_base, "images", "train")
train_lbl_dir = os.path.join(dataset_base, "labels", "train")
val_img_dir   = os.path.join(dataset_base, "images", "val")
val_lbl_dir   = os.path.join(dataset_base, "labels", "val")

os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_lbl_dir, exist_ok=True)

# Find images in the correct location
img_files = glob.glob(os.path.join(train_img_dir, "*.jpg")) + \
            glob.glob(os.path.join(train_img_dir, "*.png")) + \
            glob.glob(os.path.join(train_img_dir, "*.jpeg"))

if len(img_files) == 0:
    print("ERROR: No images found in images/train/")
    print(f"Looked in: {train_img_dir}")
    exit()

# 80/20 split
random.shuffle(img_files)
split_idx  = int(len(img_files) * 0.8)
val_images = img_files[split_idx:]

moved = 0
for img_path in val_images:
    img_name = os.path.basename(img_path)
    lbl_name = os.path.splitext(img_name)[0] + ".txt"
    lbl_path = os.path.join(train_lbl_dir, lbl_name)

    shutil.move(img_path, os.path.join(val_img_dir, img_name))
    if os.path.exists(lbl_path):
        shutil.move(lbl_path, os.path.join(val_lbl_dir, lbl_name))
    moved += 1

print(f"Split Complete.")
print(f"Train: {len(img_files) - moved} images")
print(f"Val:   {moved} images")