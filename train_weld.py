from ultralytics import YOLO
import os
import torch

DATASET_YAML = r"C:\Users\DHRUV CHAUHAN\Desktop\PROJECT\code\Weld_dataset\data.yaml"
EPOCHS = 150
RUN_NAME = "train_weld_final"

def train():
    if not os.path.exists(DATASET_YAML):
        print("ERROR: Cannot find data.yaml")
        print(f"Looked at: {DATASET_YAML}")
        return

    device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU (slow)"
    print(f"Training on: {device}")
    print("Loading base model...")

    model = YOLO('yolov8n.pt')
    print(f"Starting training for {EPOCHS} epochs...")

    model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=640,
        batch=8,
        workers=0,
        name=RUN_NAME,
        patience=30,
        plots=True,
        exist_ok=True
    )

    weights_path = os.path.join(
        r"C:\Users\DHRUV CHAUHAN\Desktop\PROJECT\code\runs\detect",
        RUN_NAME, "weights", "best.pt"
    )
    print("Training complete.")
    print(f"Model saved to: {weights_path}")

if __name__ == "__main__":
    train()