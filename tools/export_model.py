"""Export YOLOv8 weights to ONNX for use with OpenCV DNN.

Usage:
    pip install ultralytics
    python tools/export_model.py            # -> models/yolov8n.onnx
    python tools/export_model.py yolov8s    # bigger, more accurate, slower
"""
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

name = sys.argv[1] if len(sys.argv) > 1 else "yolov8n"
model = YOLO(f"{name}.pt")
# opset 12 + static 640x640 input is what OpenCV's ONNX importer handles best
exported = Path(model.export(format="onnx", opset=12, imgsz=640, simplify=True))

out_dir = Path(__file__).resolve().parent.parent / "models"
out_dir.mkdir(exist_ok=True)
shutil.move(str(exported), out_dir / exported.name)
print(f"Saved {out_dir / exported.name}")
