# Object Detection Metrics Library

## Usage

The `DetectionMetricsManager` class provides a high-level interface for calculating object detection metrics. Here's a basic usage example:

To export metrics, confusion matrix and precision x recall curves.:
```python
from detmet import compute_metrics
compute_metrics(
    groundtruth_json_path="annotations.coco.json",
    prediction_json_path="predictions.json"
)
```

### Compute Metrics Separately
To have a most granular control use the class DetectionMetricsManager. It works as an api to the DetectionMetrics class.
```python
from detmet import DetectionMetricsManager

mgr = DetectionMetricsManager(
    groundtruth_json_path="annotations.coco.json",
    prediction_json_path="predictions.json",
)
res = mgr.calculate_metrics(iou_thr=0.5, conf_thr=0.0)
res.export(format="json", output_path=".")
res.plot_pr_curves(output_path="pr.png", show=False)
res.plot_confusion_matrix(output_path="conf.png", background_class=False)
```

## Metrics Available
* Precision, Recall, F1
* Precision-Recall Curves
* Confusion Matrix ((C+background)x(C+background))
* AP per Class, mAP

## Documentation

### Input Data
The input data must be a coco groundtruth json in the format:
```
{
  "info": {
    "year": int,
    "version": str,
    "description": str,
    "contributor": str,
    "url": str, // String format
  },
  "licenses": [
    {"id": int, "name": str, "url": str}
  ],
  "categories": [
    {"id": int, "name": str, "supercategory": str}
  ],
  "images": [
    {
      "id": int,
      "file_name": str,
      "height": int,
      "width": int,
      "license": int,
      "coco_url": str
    }
  ],
  "annotations": [
    {
      "id": int,
      "image_id": int,
      "category_id": int,
      "bbox": [x, y, width, height],  // Top-left origin
      "area": float,
      "iscrowd": 0 or 1
    }
  ]
}
```
And an coco prediction json in the format:
```
[
  {
    "image_id": int,
    "category_id": int,
    "bbox": [x, y, width, height],  // Top-left origin
    "score": float 
  }
]
```
### Compute Metrics
#### ```compute_metrics``` Function

Exports metrics dictionary to JSON format, save confusion_matrix.png and pr_curves.png.
| Argument     | Type  | Default | Description                                      |
|--------------|-------|---------|--------------------------------------------------|
| `groundtruth_json_path`    | str  | -       | Path to COCO-format ground truth JSON file    |
| `prediction_json_path`| str  | -       | Path to COCO-format prediction results JSON file        |
| `iou_thr`       | float   | 0.5     | IoU threshold for true positive matching (0.0-1.0)                                 |
| `conf_thr`     | float   | 0.5  |  Confidence threshold for predictions (0.0-1.0)    |
| `exclude_classes`     | list   | None  |  List of class IDs to exclude from evaluation    |

#### ```DetectionMetricsManager``` Class

Manages the calculation of object detection metrics by comparing ground truth and prediction results.
| Argument     | Type  | Default | Description                                      |
|--------------|-------|---------|--------------------------------------------------|
| `groundtruth_json_path`    | str  | -       | Path to COCO-format ground truth JSON file    |
| `prediction_json_path`| str  | -       | Path to COCO-format prediction results JSON file        |


#### ```DetectionMetricsManager.calculate_metrics()``` Function

Calculate detection metrics for the loaded dataset.
Manages the calculation of object detection metrics by comparing ground truth and prediction results.
| Argument     | Type  | Default | Description                                      |
|--------------|-------|---------|--------------------------------------------------|
| `iou_thr`    | float  | 0.5      | IoU threshold for true positive matching (0.0-1.0)    |
| `conf_thr`| float  | 0.5       | Confidence threshold for predictions (0.0-1.0)        |
| `exclude_classes`| list  | None       |List of class IDs to exclude from evaluation,        |

Returns:
```MetricsResult```: Object containing computed metrics and visualization methods

#### ```MetricsResult.export()``` Function

Export metrics to a file in the specified format.
| Argument     | Type  | Default | Description                                      |
|--------------|-------|---------|--------------------------------------------------|
| `format`    | str  | 'json'      |  Output file format (currently only 'json' supported)    |
| `output_path`| str  | '.'       | Output directory path        |

#### ```MetricsResult.plot_pr_curves()``` Function

Plot and save Precision-Recall curves.
| Argument     | Type  | Default | Description                                      |
|--------------|-------|---------|--------------------------------------------------|
| `output_path`| str  | 'pr_curves.png'       |  Output file path for the PR curve image. If None, returns the figure without saving        |
| `show`| bool  | False       |   Whether to display the plot interactively     |
| `dpi`| int  | 100       |   Image resolution in dots per inch       |

#### ```MetricsResult.plot_confusion_matrix()``` Function

Plot and save a confusion matrix visualization.
| Argument     | Type  | Default | Description                                      |
|--------------|-------|---------|--------------------------------------------------|
| `output_path`| str  | 'confusion_matrix.png'       |  Output file path for the confusion matrix image        |
| `background_class`| bool  | False       |   Whether to include background class in the visualization     |

## Metric Calculation Methodology
### Alignment with COCO/PASCAL VOC Standards

The library follows standard object detection evaluation practices:
1. Per-Class Calculation:
    * TP/FP/FN calculated independently per class
    * Matches require IoU ≥ threshold (default=0.5)
    * Follows COCO's greedy matching algorithm
2. Crowd Handling:
    * ```iscrowd=1```  annotations excluded from matching
    * Detections matching crowd regions not penalized as FPs
3. Confidence Thresholding:
    * Predictions filtered by confidence score (default=0.5)

### Multiclass Confusion Matrix vs Detection Matrix

The library maintains two distinct confusion matrices:
| Matrix Type       | Purpose                      | Dimensions       | Background Handling                      |
|-------------------|------------------------------|------------------|------------------------------------------|
| Detection Matrix   | Calculate TP/FP/FN per class | (C+1) × (C+1)     | Last row/column = background             |
| Multiclass Matrix  | Capture class-to-class errors| (C+1) × (C+1)     | Last row/column = background             |

Key Difference:
* Detection matrix uses per-class matching (aligns with COCO standards)
* Multiclass matrix uses global matching to capture misclassifications
* This separation maintains standard TP/FP/FN calculation while providing insight into classification errors

### Optimization Trick Explanation
```
candidate_ious = iou_matrix[:, j].copy()  # IoU scores for the current detection (j)  
candidate_ious[gt_matched] = -1.0         # Mask already matched ground truths  
best_i = np.argmax(candidate_ious)        # Find the best remaining GT candidate  
best_iou = candidate_ious[best_i]         # IoU of the best match  
```
This numpy optimization efficiently finds the best unmatched ground truth:
1. Create copy of IoU scores for current prediction
2. Mask already matched ground truths with -1.0
3. Find maximum IoU among remaining candidates
4. Ensures each ground truth is only matched once
5. Vectorized operations provide significant speedup

## Key Components
### DetectionMetrics Class

Core class for metric calculation with these key methods:
| Method             | Description                                         |
|--------------------|-----------------------------------------------------|
| `process_image()`  | Processes detections for a single image             |
| `calculate_metrics()` | Computes precision, recall, F1, and mAP         |
| `_compute_map()`   | Calculates COCO-style mAP using `pycocotools`       |

### DetectionMetricsManager Class

High-level interface with these main features:
| Method              | Description                                         |
|---------------------|-----------------------------------------------------|
| `load_data()`       | Loads and processes COCO datasets                   |
| `calculate_metrics()` | Computes all metrics across all images           |
| `get_annotations()` | Retrieves annotations for specific image           |


### References
1. https://cocodataset.org/#detection-eval
2. https://github.com/sunsmarterjie/yolov12/blob/main/ultralytics/utils/metrics.py
3. https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
4. https://github.com/open-mmlab/mmdetection/blob/main/mmdet/evaluation/metrics/voc_metric.py