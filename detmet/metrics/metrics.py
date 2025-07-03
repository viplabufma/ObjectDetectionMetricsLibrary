from collections import defaultdict
from typing import Any, Dict, List
import numpy as np
from detmet.validations import validate_bounding_boxes, validate_float_and_float_like, validate_int_and_int_like, validate_normalized, validate_number_is_not_NaN, validate_number_is_not_infinite, validate_positive

def bbox_iou(box1: list[float], box2: list[float]) -> float:
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
    validate_bounding_boxes(box1, box2)
    
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
    validate_int_and_int_like(tp)
    validate_int_and_int_like(fp)
    validate_positive(tp)
    validate_positive(fp)

    # Calculate precision
    denominator = tp + fp
    return tp / denominator if denominator > 0 else 0.0


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
    validate_int_and_int_like(tp)
    validate_int_and_int_like(fn)
    validate_positive(tp)
    validate_positive(fn)    
    # Calculate recall
    denominator = tp + fn
    return tp / denominator if denominator > 0 else 0.0


def f1(precision_val: float, recall_val: float) -> float:
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
    validate_float_and_float_like(precision_val)
    validate_float_and_float_like(recall_val)
    validate_number_is_not_NaN(precision_val)
    validate_number_is_not_NaN(recall_val)
    validate_number_is_not_infinite(precision_val)
    validate_number_is_not_infinite(recall_val)
    validate_normalized(precision_val)
    validate_normalized(recall_val)
    # Calculate F1 score
    denominator = precision_val + recall_val
    return 2 * precision_val * recall_val / denominator if denominator > 0 else 0.0

def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
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
    # Type validation - allow int-like types (numpy integers, etc.)
    values = [tp, fp, fn]
    names = ['true positives', 'false positives', 'false negatives']
    
    for i, (val, name) in enumerate(zip(values, names)):
        if not isinstance(val, (int, np.integer)) if 'numpy' in globals() else not isinstance(val, int):
            try:
                values[i] = int(val)
            except (TypeError, ValueError):
                raise TypeError(f"{name.capitalize()} must be integer, got {type(val).__name__}")
    
    # Value validation
    if tp < 0:
        raise ValueError(f"True positives must be non-negative, got {tp}")
    if fp < 0:
        raise ValueError(f"False positives must be non-negative, got {fp}")
    if fn < 0:
        raise ValueError(f"False negatives must be non-negative, got {fn}")
    
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
        List of ground truths per image. Each dict must contain 'bbox' and 'category_id'
    all_preds : List[List[dict]]
        List of predictions per image. Each dict must contain 'bbox', 'score', and 'category_id'
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
    """
    # Input validation
    if not isinstance(all_gts, list) or not isinstance(all_preds, list):
        raise TypeError("all_gts and all_preds must be lists")
        
    if len(all_gts) == 0:
        raise ValueError("Groundtruth lists cannot be empty")
    
    if len(all_preds) == 0:
        raise ValueError("Predictions lists cannot be empty")
    
    # Validate IoU threshold
    if not isinstance(iou_threshold, (int, float)):
        raise TypeError("iou_threshold must be numeric")
    
    if not (0.0 <= iou_threshold <= 1.0):
        raise ValueError(f"iou_threshold must be between 0.0 and 1.0, got {iou_threshold}")
    
    if not isinstance(class_agnostic, bool):
        raise TypeError("class_agnostic must be boolean")
    
    # Validate structure of inputs
    for img_id, (gts, preds) in enumerate(zip(all_gts, all_preds)):
        if not isinstance(gts, list) or not isinstance(preds, list):
            raise TypeError(f"Image {img_id}: GTs and predictions must be lists")
        
        # Validate GT structure
        for gt_id, gt in enumerate(gts):
            if not isinstance(gt, dict):
                raise TypeError(f"Image {img_id}, GT {gt_id}: must be dict")
            
            if 'bbox' not in gt:
                raise ValueError(f"Image {img_id}, GT {gt_id}: missing 'bbox' key")
            
            if not class_agnostic and 'category_id' not in gt:
                raise ValueError(f"Image {img_id}, GT {gt_id}: missing 'category_id' key")
            
            if not isinstance(gt['bbox'], (list, tuple)) or len(gt['bbox']) != 4:
                raise ValueError(f"Image {img_id}, GT {gt_id}: 'bbox' must be list/tuple of 4 elements")
        
        # Validate prediction structure
        for pred_id, pred in enumerate(preds):
            if not isinstance(pred, dict):
                raise TypeError(f"Image {img_id}, Pred {pred_id}: must be dict")
            
            required_keys = ['bbox', 'score'] + ([] if class_agnostic else ['category_id'])
            for key in required_keys:
                if key not in pred:
                    raise ValueError(f"Image {img_id}, Pred {pred_id}: missing '{key}' key")
            
            if not isinstance(pred['bbox'], (list, tuple)) or len(pred['bbox']) != 4:
                raise ValueError(f"Image {img_id}, Pred {pred_id}: 'bbox' must be list/tuple of 4 elements")
            
            try:
                float(pred['score'])
            except (TypeError, ValueError):
                raise ValueError(f"Image {img_id}, Pred {pred_id}: 'score' must be numeric")
    
    # Early return if no predictions
    total_preds = sum(len(preds) for preds in all_preds)
    if total_preds == 0:
        return {
            'precision': np.array([]),
            'recall': np.array([]),
            'thresholds': np.array([]),
            'ap': 0.0,
            'per_class': {}
        }
    
    # Flatten and filter predictions
    flat_preds = []
    for img_id, preds in enumerate(all_preds):
        for pred in preds:
            # Skip invalid scores
            score = float(pred['score'])
            if score != score or score == float('inf') or score == float('-inf'):  # NaN or infinity
                continue
                
            flat_preds.append({
                'img_id': img_id,
                'bbox': pred['bbox'],
                'score': score,
                'class_id': 0 if class_agnostic else pred['category_id']
            })

    # Early return if no valid predictions
    if not flat_preds:
        return {
            'precision': np.array([]),
            'recall': np.array([]),
            'thresholds': np.array([]),
            'ap': 0.0,
            'per_class': {}
        }

    # Sort predictions by confidence descending
    flat_preds.sort(key=lambda x: x['score'], reverse=True)
    pred_scores = [p['score'] for p in flat_preds]

    # Prepare ground truth structure
    gt_by_class = defaultdict(list)
    total_gts = 0
    class_gt_counts = defaultdict(int)

    for img_id, gts in enumerate(all_gts):
        for gt in gts:
            # Skip crowd annotations
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

    # Early return if no ground truths
    if total_gts == 0:
        return {
            'precision': np.array([0.0] * len(flat_preds)),
            'recall': np.array([0.0] * len(flat_preds)),
            'thresholds': np.array(pred_scores),
            'ap': 0.0,
            'per_class': {}
        }

    # Initialize result storage
    precision_vals = []
    recall_vals = []
    class_data = defaultdict(lambda: {
        'tp': 0, 'fp': 0,
        'precision': [], 'recall': []
    })

    # Process predictions in descending confidence order
    for pred in flat_preds:
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
                # Skip invalid bounding boxes
                continue

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
    try:
        ap = average_precision(recall_vals, precision_vals)
    except Exception:
        ap = 0.0

    per_class_ap = {}

    # Compute per-class AP
    for class_id, data in class_data.items():
        if class_gt_counts[class_id] > 0:
            try:
                per_class_ap[class_id] = average_precision(data['recall'], data['precision'])
            except Exception:
                per_class_ap[class_id] = 0.0
        else:
            per_class_ap[class_id] = 0.0

    return {
        'precision': np.array(precision_vals),
        'recall': np.array(recall_vals),
        'thresholds': np.array(pred_scores),
        'ap': ap,
        'per_class': per_class_ap
    }


def average_precision(recall: List[float], precision: List[float]) -> float:
    """
    Compute Average Precision (AP) using 101-point interpolation (COCO standard).
    
    Parameters
    ----------
    recall : List[float]
        Recall values at different confidence thresholds
    precision : List[float]
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
    """
    
    # Type Validation
    if not isinstance(recall, list):
        raise TypeError(f"recall deve ser uma lista, recebido: {type(recall).__name__}")
    
    if not isinstance(precision, list):
        raise TypeError(f"precision deve ser uma lista, recebido: {type(precision).__name__}")
    
    # Len Validation
    if len(recall) == 0:
        raise ValueError("recall não pode ser uma lista vazia")
    
    if len(precision) == 0:
        raise ValueError("precision não pode ser uma lista vazia")
    
    if len(recall) != len(precision):
        raise ValueError(f"recall e precision devem ter o mesmo tamanho. "
                        f"recall: {len(recall)}, precision: {len(precision)}")
    
    # Validation of types of elements
    for i, r in enumerate(recall):
        if not isinstance(r, (int, float, np.number)):
            raise TypeError(f"recall[{i}] deve ser numérico, recebido: {type(r).__name__}")
        if not np.isfinite(r):
            raise ValueError(f"recall[{i}] deve ser finito, recebido: {r}")
    
    for i, p in enumerate(precision):
        if not isinstance(p, (int, float, np.number)):
            raise TypeError(f"precision[{i}] deve ser numérico, recebido: {type(p).__name__}")
        if not np.isfinite(p):
            raise ValueError(f"precision[{i}] deve ser finito, recebido: {p}")
    
    # Interval validation
    for i, r in enumerate(recall):
        if not (0.0 <= r <= 1.0):
            raise ValueError(f"recall[{i}] deve estar entre 0 e 1, recebido: {r}")
    
    for i, p in enumerate(precision):
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"precision[{i}] deve estar entre 0 e 1, recebido: {p}")
    
    # Recall order validation (must be non-descending)
    for i in range(1, len(recall)):
        if recall[i] < recall[i-1]:
            raise ValueError(f"recall deve ser não-decrescente. "
                           f"recall[{i-1}]={recall[i-1]} > recall[{i}]={recall[i]}")
    
    # Convert to numpy
    try:
        recall_np = np.array(recall, dtype=np.float64)
        precision_np = np.array(precision, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Erro ao converter listas para arrays numpy: {e}")
    
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
    
    if not np.isfinite(ap_score):
        raise ValueError(f"Resultado inválido: {ap_score}")
    
    return float(ap_score)