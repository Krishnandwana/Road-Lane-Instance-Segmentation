from ultralytics import YOLO
import torch

def main():
 
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
 
    model = YOLO("yolo11n-seg.pt")   

    # Train
    model.train(
        data="data.yaml",         
        epochs=50,
        imgsz=640,               
        batch=2,
        device=device,
        workers=2,

        # Optimization
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=5e-4,

        # Early stopping
        patience=30,

        # Augmentations
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,
        mosaic=1.0,
        mixup=0.0,

        # Segmentation
        overlap_mask=True,
        mask_ratio=1,

        # Saving
        project="runs",
        name="road_line_seg",
        save=True,
        exist_ok=True,

        # Validation
        val=True,
        plots=True
    )

if __name__ == "__main__":
    main()