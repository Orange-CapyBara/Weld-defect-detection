# pi_detect.py
# Run this script ON the Raspberry Pi 4B, not on your laptop.
# Hardware: Raspberry Pi 4B 2GB + 5MP Camera Module Rev 1.3
#
# Setup on Pi (run these first):
#   sudo apt update && sudo apt install python3-pip python3-opencv -y
#   pip3 install ultralytics
#
# Enable camera:
#   sudo raspi-config → Interface Options → Camera → Enable → Reboot

from ultralytics import YOLO
import cv2
import time
import os

# Path to NCNN model folder copied from laptop
MODEL_PATH = "/home/pi/weld_model/train_weld_4class4_ncnn_model"
CONFIDENCE = 0.25
IOU        = 0.5
CLASSES    = ['CR', 'IP']

# Pi 4B 2GB optimisation — lower resolution reduces memory pressure
# and improves FPS significantly on the weaker processor
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

def run():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: NCNN model not found at: {MODEL_PATH}")
        print("Copy the _ncnn_model folder from your laptop to the Pi first.")
        print("Command: scp -r /path/to/ncnn_model pi@<PI_IP>:/home/pi/weld_model/")
        return

    print("Loading NCNN model on Raspberry Pi 4B...")
    print("Note: Pi 4B 2GB is slower than Pi 5 — expect 3-8 FPS with NCNN")
    model = YOLO(MODEL_PATH, task='detect')
    print("Model loaded.")

    # Pi 4B uses legacy camera stack — try index 0 first
    cap = cv2.VideoCapture(0)

    # Set resolution — lower = faster on Pi 4B 2GB
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        print("Try: sudo raspi-config → Interface Options → Camera → Enable")
        print("Then reboot and run again.")
        return

    print(f"Camera opened at {FRAME_WIDTH}x{FRAME_HEIGHT}.")
    print("Press Q to quit.")

    fps_readings = []
    frame_count  = 0

    while True:
        t_start = time.time()
        ret, frame = cap.read()

        if not ret:
            print("Failed to read camera frame. Check camera connection.")
            break

        frame_count += 1

        # Run detection
        results = model.predict(
            source=frame,
            conf=CONFIDENCE,
            iou=IOU,
            verbose=False
        )

        annotated = results[0].plot()

        # FPS calculation
        elapsed = time.time() - t_start
        fps = 1.0 / elapsed if elapsed > 0 else 0
        fps_readings.append(fps)

        # Defect status overlay
        defects = results[0].boxes
        if len(defects) > 0:
            names  = [CLASSES[int(b.cls)] for b in defects if int(b.cls) < len(CLASSES)]
            conf_scores = [float(b.conf) for b in defects]
            avg_conf = sum(conf_scores) / len(conf_scores)
            status = f"DEFECT DETECTED: {', '.join(set(names))} ({avg_conf:.0%} conf)"
            color  = (0, 0, 255)
        else:
            status = "CLEAN WELD"
            color  = (0, 200, 0)

        # Overlay text
        cv2.putText(annotated, f"FPS: {fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(annotated, status,
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(annotated, f"Pi 4B | YOLOv8 Nano | NCNN",
                    (10, FRAME_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (200, 200, 200), 1)

        cv2.imshow("Weld Defect Detection — Raspberry Pi 4B", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if fps_readings:
        avg_fps = sum(fps_readings) / len(fps_readings)
        print(f"\nSession complete.")
        print(f"Frames processed: {frame_count}")
        print(f"Average FPS:      {avg_fps:.1f}")
        print(f"Average latency:  {1000/avg_fps:.0f}ms per frame")

if __name__ == "__main__":
    run()