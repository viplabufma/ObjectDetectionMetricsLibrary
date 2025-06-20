'''
Author: Matheus Levy
Organization: Viplab - UFMA
GitHub: https://github.com/viplabufma/MatheusLevy_mestrado

Module: detection_metrics.py
Description:
    Provides a comprehensive toolkit for computing object detection evaluation metrics,
    including precision, recall, F1-score, confusion matrices (binary and multiclass),
    and COCO-style mean Average Precision (mAP).
'''  
import numpy as np
from collections import defaultdict
from typing import List, Tuple
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import contextlib
import io

def bbox_iou(box1: list, box2: list) -> float:
    """
    Calculate IoU between two COCO-format bounding boxes.

    Args:
        box1 (list[float]): [x, y, width, height]
        box2 (list[float]): [x, y, width, height]

    Returns:
        float: Intersection-over-Union value between 0.0 and 1.0.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    # Coordinates of box corners
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2
    # Intersection box
    ix1, iy1 = max(x1, x2), max(y1, y2)
    ix2, iy2 = min(x1_max, x2_max), min(y1_max, y2_max)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = w1*h1 + w2*h2 - inter
    return inter/union if union>0 else 0.0

def vectorized_bbox_iou(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    """
    Vectorized IoU calculation for multiple boxes.
    Args:
        gt_boxes (np.ndarray): [N, 4] as [x1, y1, x2, y2]
        pred_boxes (np.ndarray): [M, 4] as [x1, y1, x2, x2]
    Returns:
        np.ndarray: [N, M] IoU matrix.
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
    gt_boxes = gt_boxes[:, None, :]   # shape (N, 1, 4)
    pred_boxes = pred_boxes[None, :, :]   # shape (1, M, 4)
    
    # Intersection coordinates
    x1_inter = np.maximum(gt_boxes[..., 0], pred_boxes[..., 0])
    y1_inter = np.maximum(gt_boxes[..., 1], pred_boxes[..., 1])
    x2_inter = np.minimum(gt_boxes[..., 2], pred_boxes[..., 2])
    y2_inter = np.minimum(gt_boxes[..., 3], pred_boxes[..., 3])
    
    # Intersection area
    inter_width = np.maximum(0, x2_inter - x1_inter)
    inter_height = np.maximum(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height
    
    # Union area
    gt_area = (gt_boxes[..., 2] - gt_boxes[..., 0]) * (gt_boxes[..., 3] - gt_boxes[..., 1])
    pred_area = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
    union_area = gt_area + pred_area - inter_area
    
    return inter_area / np.maximum(union_area, 1e-7)

def compute_iou_matrix(gt_anns: list, pred_anns: list) -> np.ndarray:
    """
    Build pairwise IoU matrix between ground truths and predictions.
    Returns:
        np.ndarray: IoU values of shape (num_gt, num_pred).
    """
    if not gt_anns and not pred_anns:
        return np.zeros((0,0))
    
    # Converter para arrays de bboxes
    gt_boxes = np.array([g['bbox'] for g in gt_anns], dtype=np.float32)
    if gt_boxes.size > 0:
        # Converter [x,y,w,h] para [x1,y1,x2,y2] in-place
        gt_boxes[:, 2] += gt_boxes[:, 0]  # x2 = x + w
        gt_boxes[:, 3] += gt_boxes[:, 1]  # y2 = y + h
    else:
        gt_boxes = np.zeros((0,4), dtype=np.float32)
    
    pred_boxes = np.array([p['bbox'] for p in pred_anns], dtype=np.float32)
    if pred_boxes.size > 0:
        pred_boxes[:, 2] += pred_boxes[:, 0]  # x2 = x + w
        pred_boxes[:, 3] += pred_boxes[:, 1]  # y2 = y + h
    else:
        pred_boxes = np.zeros((0,4), dtype=np.float32)
    
    return vectorized_bbox_iou(gt_boxes, pred_boxes)

def precision(tp: int, fp: int) -> float:
    """
    Compute precision: TP / (TP + FP).
    """
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def recall(tp: int, fn: int) -> float:
    """
    Compute recall: TP / (TP + FN).
    """
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def f1(precision: float, recall: float) -> float:
    """
    Compute F1 score: 2 * precision * recall / (precision + recall).
    """
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple:
    """
    Convenience method to return (precision, recall, F1) together.
    """
    p = precision(tp, fp)
    r = recall(tp, fn)
    return p, r, f1(p, r)
    
class DetectionMetrics:
    """
    Computes comprehensive object detection evaluation metrics including:
    - Precision, Recall, F1-score
    - Confusion matrices (detection and multiclass)
    - Mean Average Precision (mAP) using COCO evaluation
    - Per-class and global metrics
    
    Key Features:
    - Supports both per-image processing and batch processing
    - Handles multiple classes and background detection
    - Flexible threshold configuration (IoU and confidence)
    - Excludes specified classes from evaluation
    - Provides both numerical metrics and visual confusion matrices
    
    Attributes:
        names (Dict[int, str]): Mapping of class IDs to class names
        iou_thr (float): IoU threshold for true positive matching (default=0.5)
        conf_thr (float): Confidence threshold for predictions (default=0.5)
        matrix (np.ndarray): Global detection confusion matrix
        multiclass_matrix (np.ndarray): Multiclass confusion matrix for visualization
        class_map (dict): Mapping from class IDs to matrix indices
        background_idx (int): Matrix index for background class
        stats (dict): Accumulated statistics across processed images
        all_preds (list): All predictions across images (for COCO evaluation)
        all_gts (list): All ground truths across images
        image_counter (int): Counter for processed images
    
    Methods:
        reset(): Resets all accumulated metrics and state
        process_image(gt_anns, pred_anns): Processes detections for a single image
        compute_metrics(): Computes final evaluation metrics
        fitness: Computes single weighted fitness score
        results_dict: Comprehensive metrics in dictionary format
    """
    def __init__(
        self,
        names: dict,
        iou_thr: float = 0.5,
        conf_thr: float = 0.5,
        gt_coco: COCO = None,
        predictions_coco: COCO = None,
        exclude_classes: list = None,
    ):
        """
        Initialize DetectionMetrics with configuration parameters.

        Args:
            names (dict): Class ID to name mapping.
            iou_thr (float): IoU threshold for true positive (default 0.5).
            conf_thr (float): Confidence threshold to filter predictions (default 0.5).
            gt_coco (COCO, optional): COCO object for ground truths (for mAP).
            predictions_coco (COCO, optional): COCO object for predictions (for mAP).
            exclude_classes (list[int], optional): Class IDs to exclude from evaluation.
        """
        self.names = names
        self.iou_thr = iou_thr
        self.conf_thr = conf_thr
        self.gt_coco = gt_coco
        self.predictions_coco = predictions_coco
        self.exclude_classes = exclude_classes or []
        self.reset()

    def reset(self) -> None:
        """
        Reset all internal accumulators and state to initial conditions.

        Clears confusion matrices, statistics, stored predictions/ground truths,
        and resets the processed image counter.
        """
        self.matrix = None
        self.multiclass_matrix = None
        self.class_map = {}
        self.background_idx = None
        self.stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'support': 0})
        self.all_preds = []
        self.all_gts = []
        self.image_counter = 0

    def _initialize_global_mapping(self) -> None:
        """
        Build class index mapping and initialize confusion matrices.

        Excludes any classes listed in `exclude_classes`, then assigns contiguous
        indices to the remaining classes, reserving the last index for background.
        
        Steps:
        1. Filter valid classes (excluding specified classes)
        2. Create sorted class list and mapping
        3. Set background index
        4. Initialize detection and multiclass confusion matrices
        """
        valid_ids = [cid for cid in self.names if cid not in self.exclude_classes]
        sorted_ids = sorted(valid_ids)
        self.class_map = {cid: idx for idx, cid in enumerate(sorted_ids)}
        self.background_idx = len(self.class_map)

        size = self.background_idx + 1
        self.matrix = np.zeros((size, size), dtype=int)
        self.multiclass_matrix = np.zeros((size, size), dtype=int)

    def update_support_number(self, gt_anns: list) -> None:
        """
        Increment support (ground truth count) for each class in this image.

        Args:
            gt_anns (list[dict]): List of ground truth annotations, each with 'category_id'.
        """
        for ann in gt_anns:
            if ann.get('iscrowd', 0) == 1:
                continue
            cid = ann['category_id']
            if cid in self.class_map:
                self.stats[cid]['support'] += 1

    def process_image(self, gt_anns: list, pred_anns: list) -> None:
        """
        Process detections and ground truths for a single image.

        Steps:
          1. Filter out excluded classes from gt and predictions.
          2. Initialize global class mapping if first image.
          3. Update class support based on ground truths.
          4. Compute and accumulate detection confusion (TP/FP/FN).
          5. Update running statistics from confusion counts.
          6. Compute and accumulate multiclass confusion for visualization.
          7. Store filtered predictions above confidence threshold for COCO mAP.

        Args:
            gt_anns (list[dict]): Ground truth annotations.
            pred_anns (list[dict]): Predicted annotations with 'score', 'category_id', 'bbox'.
        """
        # Exclude unwanted classes
        gt = [g for g in gt_anns if g['category_id'] not in self.exclude_classes]
        pr = [p for p in pred_anns if p['category_id'] not in self.exclude_classes]

        if self.matrix is None:
            self._initialize_global_mapping()

        # 3. Update support counts
        self.update_support_number(gt)

        # Calcular IoU uma vez (otimização)
        iou_mat_full = compute_iou_matrix(gt, pr)

        # 4. Confusion for detection
        det_conf = self._compute_detection_confusion(gt, pr, iou_mat_full)
        self.matrix += det_conf
        self._update_stats_from_confusion(det_conf)

        # 6. Multiclass confusion
        multi_conf = self._compute_multiclass_confusion(gt, pr, iou_mat_full)
        self.multiclass_matrix += multi_conf

        # 7. Store predictions for COCO mAP
        for p in pr:
            if p['score'] >= self.conf_thr:
                self.all_preds.append({
                    'image_id': self.image_counter,
                    'category_id': p['category_id'],
                    'bbox': p['bbox'],
                    'score': p['score']
                })
        self.image_counter += 1

    def _filter_annotations(
            self,
            gt_anns: list,
            pred_anns: list
        ) -> Tuple[List[dict], List[dict], List[dict]]:
        """
        Separate crowd/non-crowd ground truths and filter predictions by confidence.

        Args:
            gt_anns: Ground truth annotations
            pred_anns: Prediction annotations

        Returns:
            Tuple containing:
            - Non-crowd ground truths
            - Crowd ground truths
            - Predictions above confidence threshold
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
        Handle edge cases for detection confusion matrix:
        - No predictions (all non-crowd GTs are FN)
        - No ground truths (all predictions are FP)

        Args:
            gt_non_crowd: Non-crowd ground truths
            gt_crowd: Crowd ground truths
            preds: Filtered predictions
            confusion: Confusion matrix to modify

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
            confusion: Confusion matrix to update

        Returns:
            Tuple of matched status lists for:
            - Ground truths (boolean list)
            - Predictions (boolean list)
        """
        gt_matched = [False] * len(gt_non_crowd)
        pred_matched = [False] * len(preds)
        # Sort predictions by confidence (descending)
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
                # Extrair IoUs relevantes
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

        return gt_matched, pred_matched

    def _process_crowd_matches(
            self,
            gt_crowd: list,
            preds: list,
            pred_matched: List[bool]
        ) -> List[bool]:
        """
        Identify predictions matching crowd annotations to avoid FP penalization.

        Args:
            gt_crowd: Crowd ground truths
            preds: Filtered predictions
            pred_matched: Current prediction matched status

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
        Update confusion matrix for unmatched items:
        - Unmatched non-crowd GTs → FN
        - Unmatched predictions not matched to crowd → FP

        Args:
            gt_non_crowd: Non-crowd ground truths
            preds: Filtered predictions
            gt_matched: GT matched status
            pred_matched: Prediction matched status
            crowd_matched: Crowd-matched status
            confusion: Confusion matrix to modify
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
        Compute detection confusion matrix (TP/FP/FN) for current image.

        Steps:
        1. Separate crowd/non-crowd GTs and filter predictions
        2. Handle edge cases (no preds/no GTs)
        3. Compute IoU matrix
        4. Perform per-class matching
        5. Process crowd matches
        6. Update confusion for unmatched items

        Args:
            gt_anns: Ground truth annotations
            pred_anns: Prediction annotations
            iou_mat_full: Precomputed full IoU matrix (optimization)

        Returns:
            Detection confusion matrix
        """
        # Initialize matrix and process annotations
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

        # Process crowd regions and unmatched items
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
    ) -> tuple[list,bool]:
        """
        Perform global greedy matching for multiclass confusion matrix.

        Steps:
        1. Initialize matched trackers
        2. Sort predictions by confidence (descending)
        3. For each prediction:
            - Find best unmatched GT (any class)
            - If IoU meets threshold, mark as matched
            - Update confusion matrix with class correspondence
        """
        n_gt = len(gt_anns)
        n_pred = len(pred_anns)
        gt_matched = np.zeros(n_gt, dtype=bool)
        pred_matched = np.zeros(n_pred, dtype=bool)
        
        # Sort predictions by confidence desc
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
        Compute a multiclass confusion matrix including background for visualization.
        Uses global matching (any-to-any) and marks FN/FP on background row/col.

        Steps:
        1. Filter predictions by confidence
        2. Initialize confusion matrix
        3. Handle edge cases
        4. Compute IoU matrix
        5. Perform global greedy matching
        6. Process unmatched items
        """
        preds = [p for p in pred_anns if p['score']>= self.conf_thr]
        size = self.background_idx+1
        confusion = np.zeros((size,size), dtype=int)
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
        Handle edge cases for confusion matrix calculation.
        
        Cases:
        1. No predictions: All ground truths become false negatives
        2. No ground truths: All predictions become false positives
        
        Args:
            gt_anns (list): Ground truth annotations
            pred_anns (list): Prediction annotations
            confusion (np.ndarray): Confusion matrix to update
            
        Returns:
            bool: True if edge case handled, False otherwise
        """
        if not pred_anns:
            for gt in gt_anns:
                cid = gt['category_id']
                if cid in self.class_map:
                    confusion[self.class_map[cid], self.background_idx] +=1
            return True
        if not gt_anns:
            for pr in pred_anns:
                cid = pr['category_id']
                if cid in self.class_map:
                    confusion[self.background_idx, self.class_map[cid]] +=1
            return True
        return False

    def _handle_unmatched_gts(self, gt_anns: list, gt_matched: list, confusion: np.ndarray) -> None:
        """
        Process unmatched ground truths as false negatives.
        
        Args:
            gt_anns (list): Ground truth annotations
            gt_matched (list): GT matched status
            confusion (np.ndarray): Confusion matrix to update
        """
        for i, matched in enumerate(gt_matched):
            if not matched:
                cid = gt_anns[i]['category_id']
                if cid in self.class_map:
                    confusion[self.class_map[cid], self.background_idx] +=1

    def _handle_unmatched_preds(self, pred_anns: list, pred_matched: list, confusion: np.ndarray) -> None:
        """
        Process unmatched predictions as false positives.
        
        Args:
            pred_anns (list): Prediction annotations
            pred_matched (list): Prediction matched status
            confusion (np.ndarray): Confusion matrix to update
        """
        for j, matched in enumerate(pred_matched):
            if not matched:
                cid = pred_anns[j]['category_id']
                if cid in self.class_map:
                    confusion[self.background_idx, self.class_map[cid]] +=1

    def _update_stats_from_confusion(self, confusion: np.ndarray) -> None:
        """
        Update accumulated statistics from detection confusion matrix.
        
        For each class:
        - TP: Diagonal elements
        - FP: Column sum minus diagonal
        - FN: Background column
        
        Args:
            confusion (np.ndarray): Detection confusion matrix
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
        Calculate final evaluation metrics for all processed images.

        This includes per-class precision, recall, F1, support, TP, FP, FN and global aggregates.
        If COCO ground truths and predictions were provided, also computes mAP metrics.

        Returns:
            dict: Nested metrics by class ID and 'global' key.
        """
        def has_coco_gt(): return self.gt_coco is not None
        def has_coco_pred(): return self.predictions_coco is not None

        # Per-class and global accumulators
        metrics = {}
        total_tp = total_fp = total_fn = total_support = 0
        for cid, idx in self.class_map.items():
            tp = self.stats[cid]['tp']
            fp = self.stats[cid]['fp']
            fn = self.stats[cid]['fn']
            sup = self.stats[cid]['support']
            p, r, f = precision_recall_f1(tp, fp, fn)
            metrics[cid] = {'precision': p, 'recall': r, 'f1': f,
                            'support': sup, 'tp': tp, 'fp': fp, 'fn': fn}
            total_tp += tp; total_fp += fp; total_fn += fn; total_support += sup

        # Global metrics
        gp, gr, gf = precision_recall_f1(
            total_tp, total_fp, total_fn)
        metrics['global'] = {'precision': gp, 'recall': gr, 'f1': gf,
                             'support': total_support, 'tp': total_tp,
                             'fp': total_fp, 'fn': total_fn}

        # COCO mAP if available
        if has_coco_gt() and has_coco_pred():
            gmap, gmap50, gmap75 = self._compute_map(self.gt_coco, self.predictions_coco)
            metrics['global']['mAP'] = float(gmap)
            metrics['global']['mAP50'] = float(gmap50)
            metrics['global']['mAP75'] = float(gmap75)
        return metrics

    def _compute_map(self, gt_coco: COCO, predictions_coco: COCO) -> tuple:
        """
        Compute COCO-style mean Average Precision (mAP) using pycocotools.

        Summarizes metrics at IoU thresholds: [.50:.95], .50, .75.

        Returns:
            tuple: (mAP@[.50:.95], mAP@.50, mAP@.75)
        """
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            evaluator = COCOeval(gt_coco, predictions_coco, 'bbox')
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
        # stats[0]=mAP, stats[1]=mAP50, stats[2]=mAP75
        return evaluator.stats[0], evaluator.stats[1], evaluator.stats[2]

    @property
    def fitness(self) -> float:
        """
        Single-value fitness score for optimization, using global F1 score.

        Returns:
            float: F1 score for all predictions.
        """
        return self.compute_metrics()['global']['f1']

    @property
    def results_dict(self) -> dict:
        """
        Flattened results dictionary mapping metric names to values.

        Example keys: 'precision/class_name', 'recall/global', etc., plus
        'multiclass_confusion_matrix'.

        Returns:
            dict: Flat mapping of metric names to numeric values.
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
        # attach multiclass confusion matrix as list
        res['multiclass_confusion_matrix'] = self.multiclass_matrix.tolist()
        return res

    def get_confusion_matrix_labels(self) -> list:
        valid_ids = [cid for cid in self.names if cid not in self.exclude_classes]
        sorted_ids = sorted(valid_ids)
        labels = [self.names[cid] for cid in sorted_ids]
        labels.append('background')
        return labels