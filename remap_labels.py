import os
import glob

labels_dir = r"C:\Users\DHRUV CHAUHAN\Desktop\PROJECT\code\Weld_dataset\labels\train"

# Only process the main train folder, not subdatasets
label_files = glob.glob(os.path.join(labels_dir, "*.txt"))

files_modified = 0
annotations_removed = 0

for file in label_files:
    if os.path.basename(file) in ["classes.txt", "data.yaml"]:
        continue

    with open(file, 'r') as f:
        lines = f.readlines()

    new_lines = []
    modified = False

    for line in lines:
        parts = line.strip().split()
        if not parts or len(parts) < 5:
            continue

        try:
            class_id = int(parts[0])
        except ValueError:
            continue

        # Safety check — skip unexpected class IDs
        if class_id > 5:
            print(f"WARNING: Unexpected class ID {class_id} in {file} — skipping line")
            continue

        if class_id in [2, 3]:
            annotations_removed += 1
            modified = True
            continue

        if class_id == 4:
            parts[0] = '2'
            modified = True
        elif class_id == 5:
            parts[0] = '3'
            modified = True

        new_lines.append(" ".join(parts) + "\n")

    if modified:
        with open(file, 'w') as f:
            f.writelines(new_lines)
        files_modified += 1

print(f"Surgery Complete. Modified {files_modified} files, removed {annotations_removed} annotations.")