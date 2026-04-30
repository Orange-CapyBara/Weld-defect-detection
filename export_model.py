# export_model.py
# Exports the trained model to NCNN format for Raspberry Pi 5 deployment.
# Run this on your laptop after training is complete.

from ultralytics import YOLO
import os

MODEL_PATH = r"C:\Users\DHRUV CHAUHAN\Desktop\PROJECT\code\runs\detect\train_weld_4class4\weights\best.pt"

def export():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at:\n{MODEL_PATH}")
        return

    print("Loading model...")
    model = YOLO(MODEL_PATH)

    print("Exporting to NCNN format for Raspberry Pi 5...")
    model.export(format='ncnn')

    print("\nExport complete.")
    print("Copy the generated folder ending in '_ncnn_model' to your Raspberry Pi 5.")
    print("Then run pi_detect.py on the Pi.")

if __name__ == "__main__":
    export()