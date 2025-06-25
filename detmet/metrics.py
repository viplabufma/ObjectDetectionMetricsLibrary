'''
Author: Matheus Levy
Organization: Viplab - UFMA
GitHub: https://github.com/viplabufma/MatheusLevy_mestrado

Module: detection_metrics.py
Description:
    Provides a comprehensive toolkit for computing object detection evaluation metrics,
    including precision, recall, F1-score, confusion matrices (binary and multiclass),
    COCO-style mean Average Precision (mAP) and Precision x Recall curves.
'''
import numpy as np
from collections import defaultdict
from typing import Any, Dict, List, Tuple
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import contextlib
import io

def bbox_iou(box1: list, box2: list) -> float:
    """
    Calculate Intersection over Union (IoU) between two COCO-format bounding boxes.

    Args:
        box1 (list[float]): [x, y, width, height] of first bounding box
        box2 (list[float]): [x, y, width, height] of second bounding box

    Returns:
        float: IoU value between 0.0 and 1.0
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


def vectorized_bbox_iou(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    """
    Compute IoU between multiple ground truth and predicted bounding boxes.

    Args:
        gt_boxes (np.ndarray): [N, 4] array of ground truth boxes as [x1, y1, x2, y2]
        pred_boxes (np.ndarray): [M, 4] array of predicted boxes as [x1, y1, x2, y2]

    Returns:
        np.ndarray: [N, M] matrix of IoU values
    """
    # Ensure arrays are 2D: (num_boxes, 4)
    gt_boxes = gt_boxes.reshape(-1, 4)
    pred_boxes = pred_boxes.reshape(-1, 4)

    N = gt_boxes.shape[0]
    M = pred_boxes.shape[0]

    # Return zero matrix if no boxes
    if N == 0 or M == 0:
        return np.zeros((N, M))

    # Expand dimensions for broadcasting
    gt_boxes = gt_boxes[:, None, :]  # shape (N, 1, 4)
    pred_boxes = pred_boxes[None, :, :]  # shape (1, M, 4)

    # Calculate intersection coordinates
    x1_inter = np.maximum(gt_boxes[..., 0], pred_boxes[..., 0])
    y1_inter = np.maximum(gt_boxes[..., 1], pred_boxes[..., 1])
    x2_inter = np.minimum(gt_boxes[..., 2], pred_boxes[..., 2])
    y2_inter = np.minimum(gt_boxes[..., 3], pred_boxes[..., 3])

    # Calculate intersection area
    inter_width = np.maximum(0, x2_inter - x1_inter)
    inter_height = np.maximum(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    # Calculate union area
    gt_area = (gt_boxes[..., 2] - gt_boxes[..., 0]) * (gt_boxes[..., 3] - gt_boxes[..., 1])
    pred_area = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
    union_area = gt_area + pred_area - inter_area

    return inter_area / np.maximum(union_area, 1e-7)


def compute_iou_matrix(gt_anns: List[dict], pred_anns: List[dict], convert_to_xyxy: bool = True) -> np.ndarray:
    """
    Compute IoU matrix between ground truth and predicted annotations.

    Args:
        gt_anns: List of ground truth annotations, each containing 'bbox' in [x, y, w, h] or [x1, y1, x2, y2].
        pred_anns: List of predicted annotations, each containing 'bbox' in [x, y, w, h] or [x1, y1, x2, y2].
        convert_to_xyxy: If True, convert bboxes from [x, y, w, h] to [x1, y1, x2, y2]. If False, assume bboxes
            are already in [x1, y1, x2, y2].

    Returns:
        np.ndarray: IoU matrix of shape (num_gt, num_pred).

    Examples:
        >>> gt = [{'bbox': [10, 10, 20, 20]}]
        >>> pred = [{'bbox': [12, 12, 18, 18]}]
        >>> iou_matrix = compute_iou_matrix(gt, pred)
    """
    if not gt_anns and not pred_anns:
        return np.zeros((0, 0))

    gt_boxes = np.array([g['bbox'] for g in gt_anns], dtype=np.float32)
    pred_boxes = np.array([p['bbox'] for p in pred_anns], dtype=np.float32)

    if convert_to_xyxy and gt_boxes.size > 0:
        gt_boxes[:, 2] += gt_boxes[:, 0]  # x2 = x + w
        gt_boxes[:, 3] += gt_boxes[:, 1]  # y2 = y + h
    if convert_to_xyxy and pred_boxes.size > 0:
        pred_boxes[:, 2] += pred_boxes[:, 0]  # x2 = x + w
        pred_boxes[:, 3] += pred_boxes[:, 1]  # y2 = y + h

    return vectorized_bbox_iou(gt_boxes, pred_boxes)


def precision(tp: int, fp: int) -> float:
    """
    Compute precision metric.

    Args:
        tp (int): Number of true positives
        fp (int): Number of false positives

    Returns:
        float: Precision value
    """
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(tp: int, fn: int) -> float:
    """
    Compute recall metric.

    Args:
        tp (int): Number of true positives
        fn (int): Number of false negatives

    Returns:
        float: Recall value
    """
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1(precision_val: float, recall_val: float) -> float:
    """
    Compute F1 score.

    Args:
        precision_val (float): Precision value
        recall_val (float): Recall value

    Returns:
        float: F1 score
    """
    return 2 * precision_val * recall_val / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0.0


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple:
    """
    Compute precision, recall, and F1 score together.

    Args:
        tp (int): Number of true positives
        fp (int): Number of false positives
        fn (int): Number of false negatives

    Returns:
        tuple: (precision, recall, F1)
    """
    p = precision(tp, fp)
    r = recall(tp, fn)
    return p, r, f1(p, r)


def compute_precision_recall_curve(
    all_gts: List[List[dict]],
    all_preds: List[List[dict]],
    iou_threshold: float = 0.5,
    class_agnostic: bool = False
) -> Dict[str, Any]:
    """
    Compute precision-recall curve data.

    Args:
        all_gts (List[List[dict]]): List of ground truths per image
        all_preds (List[List[dict]]): List of predictions per image
        iou_threshold (float): IoU threshold for true positive
        class_agnostic (bool): Whether to ignore classes

    Returns:
        Dict[str, Any]: Precision-recall curve data
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
    ap = _compute_ap(recall_vals, precision_vals)
    per_class_ap = {}

    # Compute per-class AP
    for class_id, data in class_data.items():
        if class_gt_counts[class_id] > 0:
            per_class_ap[class_id] = _compute_ap(data['recall'], data['precision'])
        else:
            per_class_ap[class_id] = 0.0

    return {
        'precision': np.array(precision_vals),
        'recall': np.array(recall_vals),
        'thresholds': np.array(pred_scores),
        'ap': ap,
        'per_class': per_class_ap
    }


def _compute_ap(recall: list, precision: list) -> float:
    """
    Compute Average Precision (AP) using 101-point interpolation (COCO standard).

    Args:
        recall (list): Recall values
        precision (list): Precision values

    Returns:
        float: Average Precision
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


class DetectionMetrics:
    """
    Computes comprehensive object detection evaluation metrics.

    Features:
    - Precision, Recall, F1-score
    - Confusion matrices (detection and multiclass)
    - Mean Average Precision (mAP) using COCO evaluation
    - Per-class and global metrics
    - Supports per-image and batch processing
    - Handles multiple classes and background detection
    - Flexible threshold configuration (IoU and confidence)
    - Excludes specified classes from evaluation
    - Provides numerical metrics and confusion matrices

    Attributes:
        names (Dict[int, str]): Class ID to name mapping
        iou_thr (float): IoU threshold for true positive matching (default=0.5)
        conf_thr (float): Confidence threshold for predictions (default=0.5)
        matrix (np.ndarray): Global detection confusion matrix
        multiclass_matrix (np.ndarray): Multiclass confusion matrix
        class_map (dict): Mapping from class IDs to matrix indices
        background_idx (int): Matrix index for background class
        stats (dict): Accumulated statistics across processed images
        all_preds (list): All predictions across images (for COCO evaluation)
        all_gts (list): All ground truths across images
        image_counter (int): Counter for processed images
    """

    def __init__(
        self,
        names: dict,
        iou_thr: float = 0.5,
        conf_thr: float = 0.5,
        gt_coco: COCO = None,
        predictions_coco: COCO = None,
        exclude_classes: list = None,
        store_pr_data: bool = False,
        store_pr_curves: bool = True
    ):
        """
        Initialize DetectionMetrics.

        Args:
            names (dict): Class ID to name mapping
            iou_thr (float): IoU threshold for true positive (default 0.5)
            conf_thr (float): Confidence threshold for predictions (default 0.5)
            gt_coco (COCO, optional): COCO object for ground truths
            predictions_coco (COCO, optional): COCO object for predictions
            exclude_classes (list[int], optional): Class IDs to exclude
            store_pr_data (bool): Whether to store PR curve data
            store_pr_curves (bool): Whether to compute PR curves
        """
        self.names = names
        self.iou_thr = iou_thr
        self.conf_thr = conf_thr
        self.gt_coco = gt_coco
        self.predictions_coco = predictions_coco
        self.exclude_classes = exclude_classes or []
        self.store_pr_data = store_pr_data
        self.per_class_iou_sum = defaultdict(float)
        self.per_class_tp_count = defaultdict(int)
        self.per_class_union = defaultdict(float)
        self.per_class_intersection = defaultdict(float)
        self.pr_gts = []
        self.pr_preds = []
        self.store_pr_curves = store_pr_curves
        self.pr_curves = {}
        self.reset()

    def reset(self) -> None:
        """Reset all internal accumulators and state."""
        self.matrix = None
        self.multiclass_matrix = None
        self.class_map = {}
        self.background_idx = None
        self.stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'support': 0})
        self.all_preds = []
        self.all_gts = []
        self.image_counter = 0
        self.per_class_iou_sum = defaultdict(float)
        self.per_class_tp_count = defaultdict(int)
        self.per_class_union = defaultdict(float)
        self.per_class_intersection = defaultdict(float)

    def _update_iou_metrics(self, cid: int, iou: float, inter: float, gt_area: float, pred_area: float) -> None:
        """
        Update accumulated IoU metrics for a class.

        Args:
            cid (int): Class ID
            iou (float): IoU value
            inter (float): Intersection area
            gt_area (float): Ground truth area
            pred_area (float): Predicted area
        """
        union = gt_area + pred_area - inter
        self.per_class_iou_sum[cid] += iou
        self.per_class_tp_count[cid] += 1
        self.per_class_intersection[cid] += inter
        self.per_class_union[cid] += union

    def _initialize_global_mapping(self) -> None:
        """Build class index mapping and initialize confusion matrices."""
        valid_ids = [cid for cid in self.names if cid not in self.exclude_classes]
        sorted_ids = sorted(valid_ids)
        self.class_map = {cid: idx for idx, cid in enumerate(sorted_ids)}
        self.background_idx = len(self.class_map)

        size = self.background_idx + 1
        self.matrix = np.zeros((size, size), dtype=int)
        self.multiclass_matrix = np.zeros((size, size), dtype=int)

    def update_support_number(self, gt_anns: list) -> None:
        """
        Increment support count for each class.

        Args:
            gt_anns (list): List of ground truth annotations
        """
        for ann in gt_anns:
            if ann.get('iscrowd', 0) == 1:
                continue
            cid = ann['category_id']
            if cid in self.class_map:
                self.stats[cid]['support'] += 1

    def process_image(self, gt_anns: list, pred_anns: list) -> None:
        """
        Process detections for a single image.

        Steps:
        1. Filter excluded classes
        2. Initialize global mapping if needed
        3. Update class support counts
        4. Compute and accumulate confusion matrices
        5. Store predictions for COCO mAP

        Args:
            gt_anns (list): Ground truth annotations
            pred_anns (list): Predicted annotations
        """
        # Filter unwanted classes
        gt = [g for g in gt_anns if g['category_id'] not in self.exclude_classes]
        pr = [p for p in pred_anns if p['category_id'] not in self.exclude_classes]

        if self.matrix is None:
            self._initialize_global_mapping()

        # Update support counts
        self.update_support_number(gt)

        # Calculate IoU matrix
        iou_mat_full = compute_iou_matrix(gt, pr)

        # Compute detection confusion
        det_conf = self._compute_detection_confusion(gt, pr, iou_mat_full)
        self.matrix += det_conf
        self._update_stats_from_confusion(det_conf)

        # Compute multiclass confusion
        multi_conf = self._compute_multiclass_confusion(gt, pr, iou_mat_full)
        self.multiclass_matrix += multi_conf

        # Store predictions for COCO mAP
        for p in pr:
            if p['score'] >= self.conf_thr:
                self.all_preds.append({
                    'image_id': self.image_counter,
                    'category_id': p['category_id'],
                    'bbox': p['bbox'],
                    'score': p['score']
                })

        # Store data for PR curve computation
        if self.store_pr_data:
            self.pr_gts.append(gt)
            self.pr_preds.append(pr)
        self.image_counter += 1

    def _filter_annotations(
        self,
        gt_anns: list,
        pred_anns: list
    ) -> Tuple[List[dict], List[dict], List[dict]]:
        """
        Separate crowd/non-crowd GTs and filter predictions by confidence.

        Args:
            gt_anns: Ground truth annotations
            pred_anns: Prediction annotations

        Returns:
            Tuple containing:
            - Non-crowd ground truths
            - Crowd ground truths
            - Filtered predictions
        """
        gt_non_crowd = [g for g in gt_anns if g.get('iscrowd', 0) == 0]
        gt_crowd = [g for g in gt_anns if g.get('iscrowd', 0) == 1]
        preds = [p for p in pred_anns if p['score'] >= self.conf_thr]
        return gt_non_crowd, gt_crowd, preds

    def _handle_detection_conf_edge_cases(
        self,
        gt_non_crowd: list,
        gt_crowd: list,
        preds: list,
        confusion: np.ndarray
    ) -> bool:
        """
        Handle edge cases for detection confusion matrix.

        Args:
            gt_non_crowd: Non-crowd ground truths
            gt_crowd: Crowd ground truths
            preds: Filtered predictions
            confusion: Confusion matrix

        Returns:
            True if edge case handled, False otherwise
        """
        # Case 1: No predictions → all non-crowd GTs are FN
        if not preds:
            for gt in gt_non_crowd:
                cid = gt['category_id']
                if cid in self.class_map:
                    idx = self.class_map[cid]
                    confusion[idx, self.background_idx] += 1
            return True

        # Case 2: No ground truths → all preds are FP
        if not gt_non_crowd and not gt_crowd:
            for pr in preds:
                cid = pr['category_id']
                if cid in self.class_map:
                    idx = self.class_map[cid]
                    confusion[self.background_idx, idx] += 1
            return True

        return False

    def _perform_per_class_matching(
        self,
        gt_non_crowd: list,
        preds: list,
        iou_mat: np.ndarray,
        confusion: np.ndarray
    ) -> Tuple[List[bool], List[bool]]:
        """
        Match predictions to ground truths per class using IoU threshold.

        Args:
            gt_non_crowd: Non-crowd ground truths
            preds: Filtered predictions
            iou_mat: Precomputed IoU matrix
            confusion: Confusion matrix

        Returns:
            Tuple of matched status for ground truths and predictions
        """
        gt_matched = [False] * len(gt_non_crowd)
        pred_matched = [False] * len(preds)
        # Sort predictions by confidence descending
        all_pred_indices = sorted(range(len(preds)),
                                key=lambda j: preds[j]['score'],
                                reverse=True)

        for cid in set(g['category_id'] for g in gt_non_crowd):
            # Get indices for current class
            gt_idxs = [i for i, g in enumerate(gt_non_crowd)
                    if g['category_id'] == cid and not gt_matched[i]]
            pr_idxs = [j for j in all_pred_indices
                    if preds[j]['category_id'] == cid and not pred_matched[j]]

            unmatched_gt_list = gt_idxs.copy()
            for j in pr_idxs:
                if not unmatched_gt_list:
                    break
                # Extract relevant IoUs
                iou_scores = iou_mat[unmatched_gt_list, j]
                best_idx = np.argmax(iou_scores)
                best_iou = iou_scores[best_idx]
                best_gt_index = unmatched_gt_list[best_idx]

                # Record TP if match meets threshold
                if best_iou >= self.iou_thr:
                    gt_matched[best_gt_index] = True
                    pred_matched[j] = True
                    idx = self.class_map[cid]
                    confusion[idx, idx] += 1
                    unmatched_gt_list.pop(best_idx)

                    # Calculate areas and intersection
                    gt_box = gt_non_crowd[best_gt_index]['bbox']
                    pred_box = preds[j]['bbox']
                    gt_area = gt_box[2] * gt_box[3]
                    pred_area = pred_box[2] * pred_box[3]
                    inter = best_iou * (gt_area + pred_area) / (1 + best_iou) if best_iou > 0 else 0.0

                    # Update IoU metrics
                    self._update_iou_metrics(
                        cid, best_iou, inter, gt_area, pred_area
                    )
        return gt_matched, pred_matched

    def _process_crowd_matches(
        self,
        gt_crowd: list,
        preds: list,
        pred_matched: List[bool]
    ) -> List[bool]:
        """
        Identify predictions matching crowd annotations.

        Args:
            gt_crowd: Crowd ground truths
            preds: Filtered predictions
            pred_matched: Prediction matched status

        Returns:
            Updated list indicating crowd-matched predictions
        """
        crowd_matched = [False] * len(preds)
        for crowd in gt_crowd:
            for j, pred in enumerate(preds):
                if pred_matched[j] or crowd_matched[j]:
                    continue
                if crowd['category_id'] != pred['category_id']:
                    continue
                if bbox_iou(crowd['bbox'], pred['bbox']) >= self.iou_thr:
                    crowd_matched[j] = True
        return crowd_matched

    def _update_confusion_unmatched(
        self,
        gt_non_crowd: list,
        preds: list,
        gt_matched: List[bool],
        pred_matched: List[bool],
        crowd_matched: List[bool],
        confusion: np.ndarray
    ) -> None:
        """
        Update confusion matrix for unmatched items.

        Args:
            gt_non_crowd: Non-crowd ground truths
            preds: Filtered predictions
            gt_matched: GT matched status
            pred_matched: Prediction matched status
            crowd_matched: Crowd-matched status
            confusion: Confusion matrix
        """
        # Process false negatives (unmatched non-crowd GTs)
        for i, matched in enumerate(gt_matched):
            if not matched:
                cid = gt_non_crowd[i]['category_id']
                if cid in self.class_map:
                    idx = self.class_map[cid]
                    confusion[idx, self.background_idx] += 1

        # Process false positives (unmatched preds not in crowd)
        for j, matched in enumerate(pred_matched):
            if not matched and not crowd_matched[j]:
                cid = preds[j]['category_id']
                if cid in self.class_map:
                    idx = self.class_map[cid]
                    confusion[self.background_idx, idx] += 1

    def _compute_detection_confusion(
        self,
        gt_anns: list,
        pred_anns: list,
        iou_mat_full: np.ndarray = None
    ) -> np.ndarray:
        """
        Compute the detection confusion matrix (True Positives, False Positives, False Negatives).

        This method calculates the confusion matrix for object detection by matching ground truth
        annotations with predictions based on the IoU threshold. It handles edge cases such as
        no predictions or no ground truths, and accounts for crowd annotations.

        Args:
            gt_anns: List of ground truth annotations, each containing 'category_id', 'bbox', and
                optionally 'iscrowd'.
            pred_anns: List of predicted annotations, each containing 'category_id', 'bbox', and 'score'.
            iou_mat_full: Precomputed IoU matrix between ground truths and predictions. If None,
                it will be computed internally.

        Returns:
            np.ndarray: Confusion matrix of shape (n_classes + 1, n_classes + 1), where the last
                row/column represents the background class.

        Examples:
            >>> metrics = DetectionMetrics(names={1: 'person'})
            >>> gt = [{'category_id': 1, 'bbox': [10, 10, 20, 20]}]
            >>> pred = [{'category_id': 1, 'bbox': [12, 12, 18, 18], 'score': 0.9}]
            >>> confusion = metrics._compute_detection_confusion(gt, pred)
        """
        # Initialize matrix
        size = self.background_idx + 1
        confusion = np.zeros((size, size), dtype=int)
        gt_non_crowd, gt_crowd, preds = self._filter_annotations(gt_anns, pred_anns)

        # Handle edge cases
        if self._handle_detection_conf_edge_cases(gt_non_crowd, gt_crowd, preds, confusion):
            return confusion

        iou_mat = iou_mat_full[:len(gt_non_crowd)] if gt_non_crowd else np.array([])

        # Perform matching
        gt_matched, pred_matched = self._perform_per_class_matching(
            gt_non_crowd, preds, iou_mat, confusion
        )

        # Process crowd regions
        crowd_matched = self._process_crowd_matches(gt_crowd, preds, pred_matched)
        self._update_confusion_unmatched(
            gt_non_crowd, preds, gt_matched, pred_matched, crowd_matched, confusion
        )

        return confusion

    def match_detection_global(
        self,
        gt_anns: list,
        pred_anns: list,
        iou_matrix: np.ndarray,
        iou_thr: float,
        confusion: np.ndarray
    ) -> tuple[list, bool, np.ndarray]:
        """
        Perform global greedy matching for multiclass confusion matrix.

        Args:
            gt_anns: Ground truth annotations
            pred_anns: Prediction annotations
            iou_matrix: IoU matrix
            iou_thr: IoU threshold
            confusion: Confusion matrix

        Returns:
            Tuple containing:
            - Matched status for ground truths
            - Matched status for predictions
            - Updated confusion matrix
        """
        n_gt = len(gt_anns)
        n_pred = len(pred_anns)
        gt_matched = np.zeros(n_gt, dtype=bool)
        pred_matched = np.zeros(n_pred, dtype=bool)

        # Sort predictions by confidence descending
        order = sorted(range(n_pred),
                     key=lambda j: pred_anns[j]['score'],
                     reverse=True)

        for j in order:
            if pred_matched[j]:
                continue

            # Vectorized matching
            candidate_ious = iou_matrix[:, j].copy()
            candidate_ious[gt_matched] = -1.0
            best_i = np.argmax(candidate_ious)
            best_iou = candidate_ious[best_i]

            if best_iou >= iou_thr:
                gt_matched[best_i] = True
                pred_matched[j] = True
                gcls = gt_anns[best_i]['category_id']
                pcls = pred_anns[j]['category_id']
                if gcls in self.class_map and pcls in self.class_map:
                    confusion[self.class_map[gcls], self.class_map[pcls]] += 1

        return gt_matched.tolist(), pred_matched.tolist(), confusion

    def _compute_multiclass_confusion(self, gt_anns: list, pred_anns: list, iou_mat_full: np.ndarray = None) -> np.ndarray:
        """
        Compute multiclass confusion matrix.

        Args:
            gt_anns: Ground truth annotations
            pred_anns: Prediction annotations
            iou_mat_full: Precomputed IoU matrix

        Returns:
            Multiclass confusion matrix
        """
        preds = [p for p in pred_anns if p['score'] >= self.conf_thr]
        size = self.background_idx + 1
        confusion = np.zeros((size, size), dtype=int)
        if self._handle_detection_edge_cases(gt_anns, preds, confusion):
            return confusion
        iou_mat = iou_mat_full

        gt_mat, pred_mat, confusion = self.match_detection_global(
            gt_anns, preds, iou_mat, self.iou_thr, confusion
        )
        self._handle_unmatched_gts(gt_anns, gt_mat, confusion)
        self._handle_unmatched_preds(preds, pred_mat, confusion)
        return confusion

    def _handle_detection_edge_cases(
        self,
        gt_anns: list,
        pred_anns: list,
        confusion: np.ndarray
    ) -> bool:
        """
        Handle edge cases for confusion matrix.

        Args:
            gt_anns: Ground truth annotations
            pred_anns: Prediction annotations
            confusion: Confusion matrix

        Returns:
            True if edge case handled, False otherwise
        """
        if not pred_anns:
            for gt in gt_anns:
                cid = gt['category_id']
                if cid in self.class_map:
                    confusion[self.class_map[cid], self.background_idx] += 1
            return True
        if not gt_anns:
            for pr in pred_anns:
                cid = pr['category_id']
                if cid in self.class_map:
                    confusion[self.background_idx, self.class_map[cid]] += 1
            return True
        return False

    def _handle_unmatched_gts(self, gt_anns: list, gt_matched: list, confusion: np.ndarray) -> None:
        """
        Process unmatched ground truths as false negatives.

        Args:
            gt_anns: Ground truth annotations
            gt_matched: GT matched status
            confusion: Confusion matrix
        """
        for i, matched in enumerate(gt_matched):
            if not matched:
                cid = gt_anns[i]['category_id']
                if cid in self.class_map:
                    confusion[self.class_map[cid], self.background_idx] += 1

    def _handle_unmatched_preds(self, pred_anns: list, pred_matched: list, confusion: np.ndarray) -> None:
        """
        Process unmatched predictions as false positives.

        Args:
            pred_anns: Prediction annotations
            pred_matched: Prediction matched status
            confusion: Confusion matrix
        """
        for j, matched in enumerate(pred_matched):
            if not matched:
                cid = pred_anns[j]['category_id']
                if cid in self.class_map:
                    confusion[self.background_idx, self.class_map[cid]] += 1

    def _update_stats_from_confusion(self, confusion: np.ndarray) -> None:
        """
        Update statistics from detection confusion matrix.

        Args:
            confusion: Detection confusion matrix
        """
        for cid, idx in self.class_map.items():
            tp = confusion[idx, idx]
            fp = confusion[:, idx].sum() - tp
            fn = confusion[idx, self.background_idx]
            self.stats[cid]['tp'] += int(tp)
            self.stats[cid]['fp'] += int(fp)
            self.stats[cid]['fn'] += int(fn)

    def compute_metrics(self) -> dict:
        """
        Calculate final evaluation metrics.

        Returns:
            dict: Comprehensive metrics dictionary
        """
        def has_coco_gt(): return self.gt_coco is not None
        def has_coco_pred(): return self.predictions_coco is not None

        # Initialize metrics dictionary
        metrics = {}
        total_tp = total_fp = total_fn = total_support = 0

        # Reset PR curves storage
        self.pr_curves = {}

        # Compute per-class metrics
        for cid, idx in self.class_map.items():
            tp = self.stats[cid]['tp']
            fp = self.stats[cid]['fp']
            fn = self.stats[cid]['fn']
            sup = self.stats[cid]['support']
            p, r, f = precision_recall_f1(tp, fp, fn)
            metrics[cid] = {'precision': p, 'recall': r, 'f1': f,
                            'support': sup, 'tp': tp, 'fp': fp, 'fn': fn}
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_support += sup

        # Compute global metrics
        gp, gr, gf = precision_recall_f1(
            total_tp, total_fp, total_fn)
        metrics['global'] = {'precision': gp, 'recall': gr, 'f1': gf,
                             'support': total_support, 'tp': total_tp,
                             'fp': total_fp, 'fn': total_fn}

        # Compute COCO mAP if available
        if has_coco_gt() and has_coco_pred():
            gmap, gmap50, gmap75, per_class_ap = self._compute_map(self.gt_coco, self.predictions_coco)
            metrics['global']['mAP'] = float(gmap)
            metrics['global']['mAP50'] = float(gmap50)
            metrics['global']['mAP75'] = float(gmap75)

            # Add per-class AP to metrics
            for cid in self.class_map:
                ap_val = per_class_ap.get(cid, 0.0)
                metrics[cid]['ap'] = float(ap_val)

        # Compute IoU metrics
        class_ious = []
        for cid in self.class_map:
            tp_count = self.per_class_tp_count.get(cid, 0)
            iou_sum = self.per_class_iou_sum.get(cid, 0.0)
            inter = self.per_class_intersection.get(cid, 0.0)
            union = self.per_class_union.get(cid, 0.0)

            # Calculate average IoU for matched pairs
            avg_iou = iou_sum / tp_count if tp_count > 0 else 0.0

            # Calculate aggregate IoU
            agg_iou = inter / union if union > 0 else 0.0

            metrics[cid].update({
                'iou': float(avg_iou),
                'agg_iou': float(agg_iou)
            })
            class_ious.append(agg_iou)

        # Calculate mIoU
        mIoU = sum(class_ious) / len(class_ious) if class_ious else 0.0
        metrics['global']['mIoU'] = float(mIoU)

        # Compute PR curves if enabled
        if self.store_pr_curves:
            self._compute_pr_curves(metrics)
            # Include PR curves data in metrics
            metrics['pr_curves'] = self.pr_curves

        return metrics

    def _compute_pr_curves(self, metrics: dict) -> None:
        """
        Compute and store Precision-Recall curves.

        Args:
            metrics (dict): Computed metrics dictionary
        """
        if not self.gt_coco or not self.predictions_coco:
            return

        try:
            with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                # Filter out excluded classes
                cat_ids = self.gt_coco.getCatIds()
                valid_cat_ids = [cat_id for cat_id in cat_ids if cat_id not in self.exclude_classes]

                evaluator = COCOeval(self.gt_coco, self.predictions_coco, 'bbox')
                evaluator.params.catIds = valid_cat_ids
                evaluator.evaluate()
                evaluator.accumulate()

                # Handle empty evaluator stats
                if not hasattr(evaluator, 'eval') or 'precision' not in evaluator.eval:
                    return

                # Parameters for curve extraction
                aind = next(i for i, aRng in enumerate(evaluator.params.areaRngLbl)
                        if aRng == 'all')
                mind = next(i for i, mDet in enumerate(evaluator.params.maxDets)
                        if mDet == 100)

                # Global PR curve (average across classes)
                recall_thrs = evaluator.params.recThrs
                precision_global = []

                for i in range(len(recall_thrs)):
                    precisions = []
                    for k in range(len(valid_cat_ids)):
                        # Use first IoU threshold (0.5)
                        val = evaluator.eval['precision'][0, i, k, aind, mind]
                        if val > -1:
                            precisions.append(val)
                    precision_global.append(np.mean(precisions) if precisions else 0)

                # Use AP from metrics if stats empty
                ap_global = metrics['global']['mAP'] if len(evaluator.stats) == 0 else evaluator.stats[0]

                self.pr_curves['global'] = {
                    'recall': recall_thrs,
                    'precision': np.array(precision_global),
                    'ap': ap_global
                }

                # Per-class PR curves
                for k, cat_id in enumerate(valid_cat_ids):
                    precision_vals = evaluator.eval['precision'][0, :, k, aind, mind]
                    valid_mask = precision_vals > -1

                    self.pr_curves[int(cat_id)] = {
                        'recall': recall_thrs[valid_mask],
                        'precision': precision_vals[valid_mask],
                        'ap': metrics.get(int(cat_id), {}).get('ap', 0)
                    }

        except (IndexError, AttributeError, KeyError) as e:
            print(f"Error generating PR curves: {str(e)}")

    def _compute_map(self, gt_coco: COCO, predictions_coco: COCO) -> tuple:
        """
        Compute COCO-style mean Average Precision (mAP).

        Args:
            gt_coco: COCO object for ground truths
            predictions_coco: COCO object for predictions

        Returns:
            tuple: (mAP, mAP50, mAP75, per_class_ap)
        """
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            evaluator = COCOeval(gt_coco, predictions_coco, 'bbox')
            cat_ids = gt_coco.getCatIds()
            valid_cat_ids = [cat_id for cat_id in cat_ids if cat_id not in self.exclude_classes]
            evaluator.params.catIds = valid_cat_ids

            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
            gmap = evaluator.stats[0]  # mAP@[.50:.95]
            gmap50 = evaluator.stats[1]  # mAP@.50
            gmap75 = evaluator.stats[2]  # mAP@.75

        # Compute per-class AP
        per_class_ap = {}
        if hasattr(evaluator, 'eval') and 'precision' in evaluator.eval:
            # Find indices for areaRng='all' and maxDets=100
            aind = next(i for i, aRng in enumerate(evaluator.params.areaRngLbl) if aRng == 'all')
            mind = next(i for i, mDet in enumerate(evaluator.params.maxDets) if mDet == 100)
            precision = evaluator.eval['precision']

            # Compute per-class AP
            for k, cat_id in enumerate(valid_cat_ids):
                pr_array = precision[:, :, k, aind, mind]  # [T, R]
                valid_pr = pr_array[pr_array > -1]
                ap = valid_pr.mean() if valid_pr.size > 0 else 0.0
                per_class_ap[cat_id] = ap

        return gmap, gmap50, gmap75, per_class_ap

    @property
    def fitness(self) -> float:
        """
        Single-value fitness score (global F1 score).

        Returns:
            float: Global F1 score
        """
        return self.compute_metrics()['global']['f1']

    @property
    def results_dict(self) -> dict:
        """
        Flattened results dictionary.

        Returns:
            dict: Flat mapping of metric names to values
        """
        metrics = self.compute_metrics()
        res = {}
        for cid, vals in metrics.items():
            if cid == 'global':
                for k, v in vals.items():
                    res[f"{k}/global"] = v
            else:
                cls_name = self.names.get(cid, f'class_{cid}')
                for k, v in vals.items():
                    res[f"{k}/{cls_name}"] = v
        # Attach multiclass confusion matrix
        res['multiclass_confusion_matrix'] = self.multiclass_matrix.tolist()
        return res

    def get_confusion_matrix_labels(self) -> list:
        """
        Get labels for confusion matrix.

        Returns:
            list: Class names with 'background' as last element
        """
        valid_ids = [cid for cid in self.names if cid not in self.exclude_classes]
        sorted_ids = sorted(valid_ids)
        labels = [self.names[cid] for cid in sorted_ids]
        labels.append('background')
        return labels