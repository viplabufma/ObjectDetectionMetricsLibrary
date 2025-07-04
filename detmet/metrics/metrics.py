from detmet.validations import Boolean, BoundingBox, GroundTruthAnnotations, PrecisionList, PrecisionRecallResult, PredictionAnnotations, ProbabilityFloat, PositiveInteger, RecallList, validated
from collections import defaultdict
from typing import Dict
import numpy as np

@validated
def bbox_iou(box1: BoundingBox, box2: BoundingBox) -> ProbabilityFloat:
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

    Raises
    ------
    ValueError
        If boxes don't have 4 elements, or widths/heights are negative
    TypeError
        If coordinates are not numeric

    Examples
    --------
    >>> box1 = [10, 10, 20, 20]
    >>> box2 = [15, 15, 20, 20]
    >>> iou = bbox_iou(box1, box2)
    """    
    x1, y1, w1, h1 = map(float, box1)
    x2, y2, w2, h2 = map(float, box2)

    # Handle zero-area boxes
    if w1 == 0 or h1 == 0 or w2 == 0 or h2 == 0:
        return 0.0

    # Calculate intersection
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2
    
    ix1, iy1 = max(x1, x2), max(y1, y2)
    ix2, iy2 = min(x1_max, x2_max), min(y1_max, y2_max)
    
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    
    # Calculate union
    union = w1 * h1 + w2 * h2 - inter
    
    # Return IoU
    return inter / union

@validated
def precision(tp: PositiveInteger, fp: PositiveInteger) -> ProbabilityFloat:
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
        Precision value between 0.0 and 1.0

    Raises
    ------
    TypeError
        If tp or fp are not integers
    ValueError
        If tp or fp are negative

    Notes
    -----
    Precision = TP / (TP + FP)
    Returns 0.0 when denominator is zero (no predictions made)
    
    Examples
    --------
    >>> precision(10, 5)  # 10 TP, 5 FP
    0.6666666666666666
    >>> precision(0, 0)   # No predictions
    0.0
    """

    # Calculate precision
    denominator = tp + fp
    return tp / denominator if denominator > 0 else 0.0

@validated
def recall(tp: PositiveInteger, fn: PositiveInteger) -> ProbabilityFloat:
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
        Recall value between 0.0 and 1.0

    Raises
    ------
    TypeError
        If tp or fn are not integers
    ValueError
        If tp or fn are negative

    Notes
    -----
    Recall = TP / (TP + FN)
    Returns 0.0 when denominator is zero (no actual positives exist)
    
    Examples
    --------
    >>> recall(10, 5)  # 10 TP, 5 FN
    0.6666666666666666
    >>> recall(0, 0)   # No actual positives
    0.0
    """
  
    # Calculate recall
    denominator = tp + fn
    return tp / denominator if denominator > 0 else 0.0

@validated
def f1(precision_val: ProbabilityFloat, recall_val: ProbabilityFloat) -> ProbabilityFloat:
    """
    Compute F1 score from precision and recall.

    Parameters
    ----------
    precision_val : float
        Precision value (should be between 0.0 and 1.0)
    recall_val : float
        Recall value (should be between 0.0 and 1.0)

    Returns
    -------
    float
        F1 score between 0.0 and 1.0

    Raises
    ------
    TypeError
        If precision_val or recall_val are not numeric
    ValueError
        If precision_val or recall_val are negative, greater than 1.0, NaN, or infinite

    Notes
    -----
    F1 = 2 * (precision * recall) / (precision + recall)
    Returns 0.0 when denominator is zero (both precision and recall are 0)
    
    Examples
    --------
    >>> f1(0.8, 0.6)  # Good precision, moderate recall
    0.6857142857142857
    >>> f1(0.0, 0.0)  # No precision or recall
    0.0
    >>> f1(1.0, 1.0)  # Perfect precision and recall
    1.0
    """

    # Calculate F1 score
    denominator = precision_val + recall_val
    return 2 * precision_val * recall_val / denominator if denominator > 0 else 0.0

@validated
def precision_recall_f1(tp: PositiveInteger, fp: PositiveInteger, fn: PositiveInteger) -> tuple[ProbabilityFloat, ProbabilityFloat, ProbabilityFloat]:
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
    tuple[float, float, float]
        (precision, recall, F1) values

    Examples
    --------
    >>> p, r, f = precision_recall_f1(5, 2, 3)
    >>> print(f"Precision: {p:.2f}, Recall: {r:.2f}, F1: {f:.2f}")
    Precision: 0.71, Recall: 0.62, F1: 0.67
    """
    p = precision(tp, fp)
    r = recall(tp, fn)
    return p, r, f1(p, r)

@validated
def precision_recall_curve(
    all_gts: GroundTruthAnnotations,
    all_preds: PredictionAnnotations,
    iou_threshold: ProbabilityFloat = 0.5,
    class_agnostic: Boolean = False
) -> PrecisionRecallResult:
    """
    Compute precision-recall curve data for object detection evaluation.

    Parameters
    ----------
    all_gts : list[list[dict]]
        List of ground truths per image. Each dict must contain:
        - 'bbox': list[float] - [x, y, width, height]
        - 'category_id': int
        - Optional: 'iscrowd' (1 indicates crowd annotation)
    all_preds : list[list[dict]]
        List of predictions per image. Each dict must contain:
        - 'bbox': list[float] - [x, y, width, height]
        - 'score': float - Confidence score
        - 'category_id': int
    iou_threshold : float, optional
        IoU threshold for true positive, by default 0.5
    class_agnostic : bool, optional
        Whether to ignore classes during matching, by default False

    Returns
    -------
    PrecisionRecallResult
        Dictionary containing:
        - 'precision': np.ndarray[float] - Precision values at each threshold
        - 'recall': np.ndarray[float] - Recall values at each threshold
        - 'thresholds': np.ndarray[float] - Confidence thresholds
        - 'ap': float - Global Average Precision
        - 'per_class': dict[int, float] - Per-class AP values

    Raises
    ------
    TypeError
        If inputs are not lists or contain invalid data types
    ValueError
        If inputs are empty, IoU threshold is invalid, or required keys are missing

    Notes
    -----
    - Processes predictions in descending confidence order
    - Implements COCO-style 101-point interpolation for AP calculation
    - Handles both class-aware and class-agnostic evaluation
    - Excludes crowd annotations from matching
    - Per-class AP is computed using global prediction order
    - Validates inputs using detmet.validations types
    """
    def process_one_prediction(pred, gt_by_class, class_data, iou_threshold):
        """Process single prediction and update TP/FP counts."""
        class_id = pred['class_id']
        best_iou = 0.0
        best_gt_idx = -1

        # Find best unmatched GT in same image and class
        for gt_idx, gt in enumerate(gt_by_class[class_id]):
            if gt['img_id'] != pred['img_id'] or gt['matched']:
                continue

            try:
                iou = bbox_iou(gt['bbox'], pred['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            except Exception:
                continue

        # Update match status
        is_tp = best_iou >= iou_threshold
        if is_tp and best_gt_idx != -1:
            gt_by_class[class_id][best_gt_idx]['matched'] = True
            class_data[class_id]['tp'] += 1
        else:
            class_data[class_id]['fp'] += 1

    def update_metrics(class_data, class_gt_counts, total_gts):
        """Compute precision/recall for all classes and globally."""
        global_tp = 0
        global_fp = 0
        
        # Update per-class metrics
        for cid, data in class_data.items():
            tp = data['tp']
            fp = data['fp']
            gt_count = class_gt_counts.get(cid, 0)
            fn = gt_count - tp

            p_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            data['precision'].append(p_val)
            data['recall'].append(r_val)
            global_tp += tp
            global_fp += fp

        # Compute global metrics
        global_fn = total_gts - global_tp
        global_p = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
        global_r = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
        return global_p, global_r
    
    def accumulate_metrics(flat_preds, gt_by_class, class_gt_counts, total_gts, iou_threshold):
        """Orchestrate metrics accumulation over all predictions."""
        precision_vals = []
        recall_vals = []
        class_data = defaultdict(lambda: {'tp': 0, 'fp': 0, 'precision': [], 'recall': []})
        
        for pred in flat_preds:
            process_one_prediction(pred, gt_by_class, class_data, iou_threshold)
            global_p, global_r = update_metrics(class_data, class_gt_counts, total_gts)
            precision_vals.append(global_p)
            recall_vals.append(global_r)
            
        return precision_vals, recall_vals, class_data
    
    def process_predictions(all_preds, class_agnostic):
        """Flatten, filter and sort predictions"""
        flat_preds = []
        for img_id, preds in enumerate(all_preds):
            for pred in preds:
                flat_preds.append({
                    'img_id': PositiveInteger.validate(img_id),
                    'bbox': BoundingBox.validate(pred['bbox']),
                    'score': ProbabilityFloat.validate(pred['score']),
                    'class_id': 0 if class_agnostic else PositiveInteger.validate(pred['category_id'])
                })
        
        if not flat_preds:
            return [], []
        
        flat_preds.sort(key=lambda x: x['score'], reverse=True)
        pred_scores = [p['score'] for p in flat_preds]
        return flat_preds, pred_scores
    
    def process_ground_truths(all_gts, class_agnostic):
        """Prepare ground truth structure and count instances"""
        gt_by_class, class_gt_counts = defaultdict(list), defaultdict(int)
        total_gts = 0

        for img_id, gts in enumerate(all_gts):
            for gt in gts:
                if PositiveInteger.validate(gt.get('iscrowd', 0)) == 1: continue
                class_id = 0 if class_agnostic else PositiveInteger.validate(gt['category_id'])
                gt_by_class[class_id].append({
                    'img_id': PositiveInteger.validate(img_id),
                    'bbox': BoundingBox.validate(gt['bbox']),
                    'matched': False
                })
                class_gt_counts[class_id] += 1
                total_gts += 1
        return gt_by_class, class_gt_counts, total_gts

    def empty_result():
        """Return empty result structure when no predictions exist"""
        return {
            'precision': np.array([]),
            'recall': np.array([]),
            'thresholds': np.array([]),
            'ap': 0.0,
            'per_class': {}
        }
    
    def empty_result_with_length(length, pred_scores):
        """Return result structure for zero ground truths"""
        return {
            'precision': np.array([0.0] * length),
            'recall': np.array([0.0] * length),
            'thresholds': np.array(pred_scores),
            'ap': 0.0,
            'per_class': {}
    }
    def per_class_ap(class_data, class_gt_counts) -> Dict:
        """Compute per-class AP from accumulated metrics"""
        per_class_ap = {}
        for class_id, data in class_data.items():
            if class_gt_counts[class_id] > 0:
                per_class_ap[class_id] = average_precision(data['recall'], data['precision'])
            else:
                per_class_ap[class_id] = 0.0
        return per_class_ap
    
    flat_preds, pred_scores = process_predictions(all_preds, class_agnostic)
    if not flat_preds: return empty_result()
    gt_by_class, class_gt_counts, total_gts = process_ground_truths(all_gts, class_agnostic)
    if total_gts == 0: return empty_result_with_length(len(flat_preds), pred_scores)
    
    precision_vals, recall_vals, class_data = accumulate_metrics(
        flat_preds, gt_by_class, class_gt_counts, total_gts, iou_threshold
    )
    ap = average_precision(recall_vals, precision_vals)
    return {
        'precision': np.array(precision_vals),
        'recall': np.array(recall_vals),
        'thresholds': np.array(pred_scores),
        'ap': ap,
        'per_class': per_class_ap(class_data, class_gt_counts)
    }


@validated
def average_precision(recall: RecallList, precision: PrecisionList) -> ProbabilityFloat:
    """
    Compute Average Precision (AP) using 101-point interpolation (COCO standard).
    
    Parameters
    ----------
    recall : list[float]
        Recall values at different confidence thresholds
    precision : list[float]
        Precision values at different confidence thresholds
    
    Returns
    -------
    float
        Average Precision (AP) score
    
    Raises
    ------
    TypeError
        If inputs are not lists or contain non-numeric values
    ValueError
        If inputs are empty, have different lengths, or contain invalid values
    
    Notes
    -----
    - Implements COCO-style 101-point interpolation
    - Ensures precision is monotonically decreasing
    - Handles edge cases with no detections
    - Validates inputs using detmet.validations types
    """
    # Cross-length validation
    if len(recall) != len(precision):
        raise ValueError(
            f"recall and precision must have the same length. "
            f"recall: {len(recall)}, precision: {len(precision)}"
        )
    
    # Convert to numpy
    recall_np = np.array(recall, dtype=np.float64)
    precision_np = np.array(precision, dtype=np.float64)
    
    # Pad with 0 and 1 endpoints
    r = np.array([0.0] + list(recall_np) + [1.0])
    p = np.array([0.0] + list(precision_np) + [0.0])
    
    # Ensure monotonic decreasing precision
    for i in range(len(p) - 2, -1, -1):
        p[i] = max(p[i], p[i + 1])
    
    # Create 101 recall points
    r_interp = np.linspace(0, 1, 101)
    p_interp = np.interp(r_interp, r, p)
    
    # Final result Validation
    ap_score = np.mean(p_interp)
    return ProbabilityFloat.validate(ap_score)