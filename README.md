# Object Detection Metrics Library

## Get Started

### Using DetectionMetricsManager
The `DetectionMetricsManager` class provides a high-level interface for calculating object detection metrics. Here's a basic usage example:

```python
from DetectionMetricsManager import DetectionMetricsManager, export_metrics, save_confusion_matrix

# Initialize with ground truth and prediction paths
gt_path = "/path/to/gt_coco.json"
pred_path = "/path/to/predictions_coco.json"
manager = DetectionMetricsManager(gt_path=gt_path, result_path=pred_path)

# Calculate metrics
metrics = manager.calculate_metrics()

# Export metrics to JSON
export_metrics(metrics, manager.labels)

# Save confusion matrix visualization
save_confusion_matrix(
    metrics['confusion_matrix_multiclass'],
    manager.labels,
    path='confusion_matrix.png',
    background_class=True
)
```
### Key Steps:
1. Initialize with paths to COCO-format JSON files
2. Load data into memory
3. Calculate metrics using default thresholds (IoU=0.5, confidence=0.5)
4. Export metrics to JSON
5. Visualize confusion matrix

### Exporting Metrics
```export_metrics``` Function

Exports metrics dictionary to JSON format with class names instead of IDs.
| Argument     | Type  | Default | Description                                      |
|--------------|-------|---------|--------------------------------------------------|
| `metrics`    | dict  | -       | Metrics dictionary from `calculate_metrics()`    |
| `class_names`| list  | -       | List of class names in order of class IDs        |
| `path`       | str   | '.'     | Output directory                                 |
| `format`     | str   | 'json'  | Output format (currently only JSON supported)    |

```save_confusion_matrix``` Function

Saves confusion matrix as a visual heatmap image.
| Argument         | Type       | Default                 | Description                                         |
|------------------|------------|-------------------------|-----------------------------------------------------|
| `matrix`         | list[list] | -                       | 2D confusion matrix                                 |
| `class_names`    | list[str]  | -                       | Class names in matrix order                         |
| `path`           | str        | 'confusion_matrix.png'  | Output file path                                    |
| `background_class` | bool     | False                   | Whether matrix includes background class            |

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
candidate_ious = iou_matrix[:, j].copy()
candidate_ious[gt_matched] = -1.0
best_i = np.argmax(candidate_ious)
best_iou = candidate_ious[best_i]
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