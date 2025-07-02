from collections import defaultdict
from typing import Any, Dict, List
import numpy as np

def bbox_iou(box1: list, box2: list) -> float:
    """
    Calculate Intersection over Union (IoU) between two COCO-format bounding boxes.

    Parameters
    ----------
    box1 : list[float]
        [x, y, width, height] of first bounding box
    box2 : list[float]
        [x, y, width, height] of second bounding box

    Returns
    -------
    float
        IoU value between 0.0 and 1.0

    Examples
    --------
    >>> box1 = [10, 10, 20, 20]
    >>> box2 = [15, 15, 20, 20]
    >>> iou = bbox_iou(box1, box2)
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    # Calculate corner coordinates
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2
    # Calculate intersection coordinates
    ix1, iy1 = max(x1, x2), max(y1, y2)
    ix2, iy2 = min(x1_max, x2_max), min(y1_max, y2_max)
    # Calculate intersection area
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def precision(tp: int, fp: int) -> float:
    """
    Compute precision metric.

    Parameters
    ----------
    tp : int
        Number of true positives
    fp : int
        Number of false positives

    Returns
    -------
    float
        Precision value

    Notes
    -----
    Precision = TP / (TP + FP)
    Returns 0.0 when denominator is zero
    """
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(tp: int, fn: int) -> float:
    """
    Compute recall metric.

    Parameters
    ----------
    tp : int
        Number of true positives
    fn : int
        Number of false negatives

    Returns
    -------
    float
        Recall value

    Notes
    -----
    Recall = TP / (TP + FN)
    Returns 0.0 when denominator is zero
    """
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1(precision_val: float, recall_val: float) -> float:
    """
    Compute F1 score from precision and recall.

    Parameters
    ----------
    precision_val : float
        Precision value
    recall_val : float
        Recall value

    Returns
    -------
    float
        F1 score

    Notes
    -----
    F1 = 2 * (precision * recall) / (precision + recall)
    Returns 0.0 when denominator is zero
    """
    return 2 * precision_val * recall_val / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0.0


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple:
    """
    Compute precision, recall, and F1 score together.

    Parameters
    ----------
    tp : int
        Number of true positives
    fp : int
        Number of false positives
    fn : int
        Number of false negatives

    Returns
    -------
    tuple
        (precision, recall, F1) values

    Examples
    --------
    >>> p, r, f = precision_recall_f1(5, 2, 3)
    """
    p = precision(tp, fp)
    r = recall(tp, fn)
    return p, r, f1(p, r)


def precision_recall_curve(
    all_gts: List[List[dict]],
    all_preds: List[List[dict]],
    iou_threshold: float = 0.5,
    class_agnostic: bool = False
) -> Dict[str, Any]:
    """
    Compute precision-recall curve data for object detection evaluation.

    Parameters
    ----------
    all_gts : List[List[dict]]
        List of ground truths per image
    all_preds : List[List[dict]]
        List of predictions per image
    iou_threshold : float, optional
        IoU threshold for true positive, by default 0.5
    class_agnostic : bool, optional
        Whether to ignore classes during matching, by default False

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'precision': Array of precision values
        - 'recall': Array of recall values
        - 'thresholds': Array of confidence thresholds
        - 'ap': Global Average Precision
        - 'per_class': Dict of per-class AP values

    Notes
    -----
    - Processes predictions in descending confidence order
    - Implements COCO-style 101-point interpolation for AP calculation
    - Handles both class-aware and class-agnostic evaluation
    - Excludes crowd annotations from matching
    """
    # Flatten and filter predictions
    flat_preds = []
    for img_id, preds in enumerate(all_preds):
        for p in preds:
            flat_preds.append({
                'img_id': img_id,
                'bbox': p['bbox'],
                'score': p['score'],
                'class_id': 0 if class_agnostic else p['category_id']
            })

    # Sort predictions by confidence descending
    flat_preds.sort(key=lambda x: x['score'], reverse=True)
    pred_scores = [p['score'] for p in flat_preds]

    # Prepare ground truth structure
    gt_by_class = defaultdict(list)
    total_gts = 0
    class_gt_counts = defaultdict(int)

    for img_id, gts in enumerate(all_gts):
        for gt in gts:
            if gt.get('iscrowd', 0) == 1:
                continue

            class_id = 0 if class_agnostic else gt['category_id']
            gt_by_class[class_id].append({
                'img_id': img_id,
                'bbox': gt['bbox'],
                'matched': False  # Track matching status
            })
            class_gt_counts[class_id] += 1
            total_gts += 1

    # Initialize result storage
    precision_vals = []
    recall_vals = []
    class_data = defaultdict(lambda: {
        'tp': 0, 'fp': 0,
        'precision': [], 'recall': []
    })

    # Process predictions in descending confidence order
    for _, pred in enumerate(flat_preds):
        class_id = pred['class_id']
        best_iou = 0.0
        best_gt_idx = -1

        # Find best unmatched GT in same image and class
        for gt_idx, gt in enumerate(gt_by_class[class_id]):
            if gt['img_id'] != pred['img_id'] or gt['matched']:
                continue

            iou = bbox_iou(gt['bbox'], pred['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        # Update match status
        is_tp = best_iou >= iou_threshold
        if is_tp and best_gt_idx != -1:
            gt_by_class[class_id][best_gt_idx]['matched'] = True
            class_data[class_id]['tp'] += 1
        else:
            class_data[class_id]['fp'] += 1

        # Compute precision/recall for each class
        for cid, data in class_data.items():
            tp = data['tp']
            fp = data['fp']
            fn = class_gt_counts[cid] - tp

            p_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            data['precision'].append(p_val)
            data['recall'].append(r_val)

        # Compute global precision/recall
        global_tp = sum(data['tp'] for data in class_data.values())
        global_fp = sum(data['fp'] for data in class_data.values())
        global_fn = total_gts - global_tp

        global_p = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
        global_r = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0

        precision_vals.append(global_p)
        recall_vals.append(global_r)

    # Compute AP with proper interpolation
    ap = average_precision(recall_vals, precision_vals)
    per_class_ap = {}

    # Compute per-class AP
    for class_id, data in class_data.items():
        if class_gt_counts[class_id] > 0:
            per_class_ap[class_id] = average_precision(data['recall'], data['precision'])
        else:
            per_class_ap[class_id] = 0.0

    return {
        'precision': np.array(precision_vals),
        'recall': np.array(recall_vals),
        'thresholds': np.array(pred_scores),
        'ap': ap,
        'per_class': per_class_ap
    }


def average_precision(recall: list, precision: list) -> float:
    """
    Compute Average Precision (AP) using 101-point interpolation (COCO standard).

    Parameters
    ----------
    recall : list
        Recall values at different confidence thresholds
    precision : list
        Precision values at different confidence thresholds

    Returns
    -------
    float
        Average Precision (AP) score

    Notes
    -----
    - Implements COCO-style 101-point interpolation
    - Ensures precision is monotonically decreasing
    - Handles edge cases with no detections
    """
    # Pad with 0 and 1 endpoints
    r = np.array([0.0] + recall + [1.0])
    p = np.array([0.0] + precision + [0.0])

    # Ensure monotonic decreasing precision
    for i in range(len(p) - 2, -1, -1):
        p[i] = max(p[i], p[i + 1])

    # Create 101 recall points
    r_interp = np.linspace(0, 1, 101)
    p_interp = np.interp(r_interp, r, p)

    return np.mean(p_interp)