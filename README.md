# detmet

A lightweight Python library for computing object detection metrics (mAP, PR curves, confusion matrices) using COCO-format ground truth and predictions.

## Features

- **High-level API**: `DetectionMetricsManager` for quick evaluation from JSON files.
- **Low-level API**: `DetectionMetrics` for custom datasets and fine-grained control.
- **Standard Metrics**: Precision, Recall, F1, Support, mAP, mAP50, mAP75, mIoU (global) and AP/IoU per class.
- **Visualizations**: Precision-Recall curves and confusion matrices.
- **Export**: Metrics to JSON.
- **COCO Compatible**: Exact input format matching COCO annotations and predictions.
- **Flexible**: Configurable IoU thresholds, confidence thresholds, class exclusions.

## Installation

```bash
pip install detmet
```

## Quick Start

```python
from detmet import DetectionMetricsManager

mgr = DetectionMetricsManager(
    groundtruth_json_path="path/to/coco_annotations.json",
    prediction_json_path="path/to/predictions.json"
)

res = mgr.calculate_metrics(
    iou_thr=0.5,
    conf_thr=0.0,
    exclude_classes=None
)

# Inspect metrics
print(res.metrics["global"]["mAP50"])

# Export to JSON (writes metrics.json)
res.export(format="json", output_path=".")

# Plot PR curves
res.plot_pr_curves(output_path="pr_curves.png", show=False)

# Plot confusion matrix
res.plot_confusion_matrix(output_path="confusion_matrix.png", background_class=False)
```

### One-line helper

```python
from detmet import compute_metrics

compute_metrics(
    groundtruth_json_path="annotations.coco.json",
    prediction_json_path="predictions.json",
    iou_thr=0.5,
    conf_thr=0.0
)
```

This writes `metrics.json`, `confusion_matrix.png`, and `pr_curves.png` to the current directory.

## Input Data Format

### Ground Truth (COCO Annotations JSON)

```json
{
  "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
  "images": [{"id": 0, "file_name": "image.jpg", "height": 480, "width": 640}],
  "annotations": [
    {
      "id": 1,
      "image_id": 0,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": 1024,
      "iscrowd": 0
    }
  ]
}
```

### Predictions JSON

```json
[
  {
    "image_id": 0,
    "category_id": 1,
    "bbox": [x, y, width, height],
    "score": 0.95
  }
]
```

Images must be referenced by `image_id`, categories by `category_id`. Bounding boxes use `[x_min, y_min, width, height]`.

## High-Level Usage: `DetectionMetricsManager`

Load COCO JSON files and compute metrics in one call:

```python
from detmet import DetectionMetricsManager

mgr = DetectionMetricsManager("gt.json", "pred.json")
res = mgr.calculate_metrics(iou_thr=0.5, conf_thr=0.0)
```

- `iou_thr`: IoU threshold for matching (default 0.5).
- `conf_thr`: Minimum confidence to consider predictions (default 0.5).
- `exclude_classes`: List of category IDs to ignore (dataset-specific).

## Low-Level Usage: `DetectionMetrics`

For in-memory data or custom processing:

```python
from detmet import DetectionMetrics

dm = DetectionMetrics(names={1: "person", 2: "car"}, iou_thr=0.5, conf_thr=0.0)

# For each image
dm.process_image(gt_annotations, pred_annotations)

metrics_dict = dm.compute_metrics()
print(metrics_dict["global"]["precision"])
print(metrics_dict[1]["recall"])  # class id 1
```

Note: COCO-style mAP and PR curves are computed when COCO objects are available (the manager provides them). For list-based usage you still get precision/recall/F1 and confusion matrices.

## Inspecting Metrics

The `MetricsResult` object exposes a metrics dictionary:

```python
print(res.metrics["global"]["mAP"])
print(res.metrics["global"]["mAP50"])
print(res.metrics["person"]["precision"])  # class name from COCO categories
```

`res.metrics` is a dict. Convert to a DataFrame if needed:

```python
import pandas as pd

df = pd.DataFrame.from_dict(res.metrics, orient="index")
```

## Visualizations

```python
# Precision-Recall curves per class
res.plot_pr_curves(output_path="pr.png", show=False)

# Confusion matrix (counts with background)
res.plot_confusion_matrix(output_path="conf.png", background_class=False)
```

## Tutorial Notebook

Full examples and advanced usage:

[examples/tutorial.ipynb](examples/tutorial.ipynb)

## Methodology Notes

- **Matching**: Predictions matched to GT using IoU >= threshold with greedy matching.
- **mAP**: Computed by COCOeval (mAP@[.50:.95], mAP50, mAP75) when COCO objects are available.
- **PR Curves**: Extracted from COCOeval data when enabled.
- **Confusion Matrices**: Count-based matrices with a background row/column.

## Key Components

- **`DetectionMetricsManager`**: Loads JSON, parses, instantiates `DetectionMetrics`.
- **`DetectionMetrics`**: Core class for metric computation from lists of dicts.
- **`MetricsResult`**: Stores metrics dict, plots, and export methods.

## FAQ

- **Why is `mAP` missing in low-level usage?** `DetectionMetrics` computes COCO mAP when COCO objects are provided (e.g., via `DetectionMetricsManager`). With list-based usage you still get precision/recall/F1 and confusion matrices.
- **Which IDs go in `exclude_classes`?** Use the dataset `category_id` values. There is no implicit background class ID to exclude.
- **Why do class keys appear as names in `res.metrics`?** The manager maps class IDs to category names for readability; use `res.metrics["global"]` for global metrics.
- **What files does `compute_metrics` write?** It writes `metrics.json`, `confusion_matrix.png`, and `pr_curves.png` to the output directory.

## References

- https://cocodataset.org/#detection-eval
- https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py