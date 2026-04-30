from ultralytics import YOLO
import os

MODEL_PATH   = r"C:\Users\DHRUV CHAUHAN\Desktop\PROJECT\code\runs\detect\train_weld_4class4\weights\best.pt"
DATASET_YAML = r"C:\Users\DHRUV CHAUHAN\Desktop\PROJECT\code\Weld_dataset\data.yaml"
CLASSES      = ['CR', 'IP']

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at:\n{MODEL_PATH}")
        return

    if not os.path.exists(DATASET_YAML):
        print(f"ERROR: data.yaml not found at:\n{DATASET_YAML}")
        return

    print("Loading model...")
    model = YOLO(MODEL_PATH)

    print("Running validation...")
    metrics = model.val(
        data=DATASET_YAML,
        split='val',
        batch=8,
        workers=0,
        plots=True
    )

    print("\n" + "=" * 40)
    print("OVERALL PERFORMANCE")
    print("=" * 40)
    print(f"mAP50:       {metrics.box.map50:.4f}  ({metrics.box.map50*100:.1f}%)")
    print(f"mAP50-95:    {metrics.box.map:.4f}  ({metrics.box.map*100:.1f}%)")
    print(f"Precision:   {metrics.box.mp:.4f}  ({metrics.box.mp*100:.1f}%)")
    print(f"Recall:      {metrics.box.mr:.4f}  ({metrics.box.mr*100:.1f}%)")

    print("\n" + "=" * 40)
    print("PER CLASS BREAKDOWN")
    print("=" * 40)
    for i, name in enumerate(CLASSES):
        try:
            print(f"{name}: AP50={metrics.box.ap50[i]:.4f}  "
                  f"P={metrics.box.p[i]:.4f}  "
                  f"R={metrics.box.r[i]:.4f}")
        except:
            print(f"{name}: N/A")
    print("=" * 40)

if __name__ == '__main__':
    main()