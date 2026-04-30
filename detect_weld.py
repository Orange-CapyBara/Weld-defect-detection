# detect_weld.py
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
import os
import sys

MODEL_PATH  = r"C:\Users\DHRUV CHAUHAN\Desktop\PROJECT\code\runs\detect\train_weld_4class4\weights\best.pt"
SAVE_DIR    = r"C:\Users\DHRUV CHAUHAN\Desktop\PROJECT\code\runs\detect\predictions"
CONFIDENCE  = 0.25
IOU         = 0.5

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at:\n{MODEL_PATH}")
        sys.exit()
    print("Loading model... please wait.")
    return YOLO(MODEL_PATH)

def webcam_mode(model):
    print("\nStarting webcam...")
    print("Press Q on the video window to quit.\n")
    model.predict(
        source="0",
        show=True,
        conf=CONFIDENCE,
        iou=IOU,
        line_width=2
    )

def image_mode(model):
    print("\nOpening file selector...")
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select a weld image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
    )
    root.destroy()

    if not path:
        print("No file selected.")
        return

    print(f"Running detection on: {path}\n")
    results = model.predict(
        source=path,
        show=True,
        conf=CONFIDENCE,
        iou=IOU,
        save=True,
        project=SAVE_DIR,
        name="predict",
        exist_ok=True,
        line_width=2
    )

    detections = len(results[0].boxes)
    if detections == 0:
        print("Result: CLEAN WELD — no defects detected above threshold.")
    else:
        names = [results[0].names[int(b.cls)] for b in results[0].boxes]
        print(f"Result: {detections} defect(s) detected — {', '.join(names)}")
    print(f"Image saved to: {SAVE_DIR}")

def folder_mode(model):
    print("\nOpening folder selector...")
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select folder containing weld images")
    root.destroy()

    if not folder:
        print("No folder selected.")
        return

    print(f"Running detection on folder: {folder}\n")
    model.predict(
        source=folder,
        conf=CONFIDENCE,
        iou=IOU,
        save=True,
        project=SAVE_DIR,
        name="predict",
        exist_ok=True,
        line_width=2
    )
    print(f"All results saved to: {SAVE_DIR}")

def main():
    print("=" * 45)
    print("     WELD DEFECT DETECTION SYSTEM")
    print("     Classes: CR (Crack), IP (Incomplete Penetration)")
    print("=" * 45)

    model = load_model()
    print("Model loaded successfully.\n")

    print("Select detection mode:")
    print("  1 - Webcam (real-time detection)")
    print("  2 - Single image (file picker)")
    print("  3 - Batch images (folder picker)")
    print("  4 - Exit")

    choice = input("\nEnter choice (1/2/3/4): ").strip()

    if choice == "1":
        webcam_mode(model)
    elif choice == "2":
        image_mode(model)
    elif choice == "3":
        folder_mode(model)
    elif choice == "4":
        print("Exiting.")
    else:
        print(f"Invalid choice '{choice}'. Defaulting to webcam.")
        webcam_mode(model)

if __name__ == "__main__":
    main()