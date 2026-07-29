# YOLO Road Lane Instance Segmentation

Instance segmentation of road surfaces and lane-line markings using a fine-tuned
YOLO11n-seg model, trained on a custom COCO-format dataset and evaluated on
an out-of-distribution video sourced from the internet.

## Demo

<video src="output.mp4" controls width="700"></video>

## Training Results

<p align="center">
  <img src="runs/road_line_seg/results.png" width="800" alt="Training curves">
</p>

<p align="center">
  <img src="runs/road_line_seg/val_batch2_pred.jpg" width="800" alt="Validation batch predictions">
</p>

<details>
<summary>More plots (PR curves, confusion matrix)</summary>
<br>

<p align="center">
  <img src="runs/road_line_seg/BoxPR_curve.png" width="400" alt="Box PR curve">
  <img src="runs/road_line_seg/MaskPR_curve.png" width="400" alt="Mask PR curve">
  <img src="runs/road_line_seg/confusion_matrix_normalized.png" width="400" alt="Confusion matrix">
</p>

</details>

*Epoch 50/50 — full run artifacts under `runs/road_line_seg/`.*

## Classes

The model is trained to segment 7 classes related to road lane markings:

| ID | Class          |
|----|----------------|
| 0  | road-roads     |
| 1  | divider-line   |
| 2  | dotted-line    |
| 3  | double-line    |
| 4  | random-line    |
| 5  | road-sign-line |
| 6  | solid-line     |

## Dataset

- Source: [Road Lane Instance Segmentation](https://www.kaggle.com/datasets/sovitrath/road-lane-instance-segmentation/data) (Kaggle)
- Format: COCO instance segmentation (`instances_train2017.json`, etc.)
- Converted to YOLO segmentation format via `ultralytics.data.converter.convert_coco` (see `main.ipynb`)
- 1021 training images, 9265 annotations
- Final dataset layout defined in `data.yaml`, rooted at `coco_converted/`

The dataset itself is **not included** in this repo — download it from the
Kaggle link above and run the conversion steps in `main.ipynb` to reproduce
`coco_converted/`.

```
coco_converted/
├── images/{train2017,val2017}
└── labels/{train2017,val2017}
```

## Training

Training is done in `train.py`, starting from the pretrained `yolo11n-seg.pt`
checkpoint.

```bash
python train.py
```

Key settings:

| Parameter        | Value                    |
|------------------|--------------------------|
| Base weights     | `yolo11n-seg.pt`         |
| Epochs           | 50                       |
| Image size       | 640                      |
| Batch size       | 2                        |
| Optimizer        | AdamW (lr0 = 0.001, weight_decay = 5e-4) |
| Patience         | 30                       |
| Augmentations    | HSV jitter, ±5° rotation, 0.1 translate, 0.5 scale, horizontal flip, mosaic |
| Mask settings    | `overlap_mask=True`, `mask_ratio=1` |

Outputs (weights, curves, confusion matrices, batch previews) are saved to
`runs/road_line_seg/`. See **Training Results** above for the headline metrics.

## Inference

`inference.py` runs the trained model on a video, draws segmentation masks,
and overlays a live FPS counter, writing the annotated result to a new video
file.

```bash
python inference.py
```

| Setting      | Value                                 |
|--------------|----------------------------------------|
| Model        | `runs/road_line_seg/weights/best.pt`   |
| Input video  | `test.mp4`                              |
| Output video | `output.mp4`                            |
| Confidence   | 0.4                                      |

## Evaluation

`test.mp4` was picked at random from the internet — i.e. it is **not** part
of the training/validation distribution and has no ground-truth labels. To
get a qualitative read on real-world generalization, the annotated output
(`output.mp4`) was reviewed by GPT and scored across several qualitative
dimensions. This is a subjective, non-benchmarked assessment (no IoU/ground
truth involved) and should be treated as a rough sanity check rather than a
formal metric.

**Overall score: 8.7 / 10 (87 / 100)**

| Category                | Score   |
|--------------------------|---------|
| Detection Accuracy         | 8.8 / 10 |
| Bounding Box Quality       | 8.5 / 10 |
| Temporal Stability         | 9.0 / 10 |
| False Positives            | 9.2 / 10 |
| False Negatives            | 8.2 / 10 |
| Confidence Calibration      | 8.5 / 10 |
| Production Readiness       | 8.8 / 10 |

## Repo structure

```
.
├── data.yaml              # Dataset config (classes + paths)
├── train.py                # Training script
├── inference.py             # Video inference script
├── main.ipynb                # COCO -> YOLO dataset conversion notebook
├── yolo11n-seg.pt              # Pretrained base weights
├── coco_converted/               # Converted YOLO-format dataset
├── runs/road_line_seg/              # Training run outputs (weights, curves, plots)
├── test.mp4                           # Random internet video used for inference
└── output.mp4                          # Annotated inference output
```

## Requirements

- Python 3.x
- `ultralytics`
- `opencv-python`
- `torch` (CUDA optional, falls back to CPU)
