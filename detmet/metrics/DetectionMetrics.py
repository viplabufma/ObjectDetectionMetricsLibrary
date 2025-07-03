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
from detmet.metrics import bbox_iou, precision_recall_f1
from detmet.utils import compute_iou_matrix
import numpy as np
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import contextlib
import io


class PrecisionRecallConfig:
    """
    Configuration for Precision-Recall curve computation and storage.
    
    Parameters
    ----------
    store_pr_data : bool, optional
        Whether to store raw data for PR curve computation, by default False
    store_pr_curves : bool, optional
        Whether to compute and store PR curves, by default True
    """
    
    def __init__(self, store_pr_data: bool = False, store_pr_curves: bool = True):
        self.store_pr_data = store_pr_data
        self.store_pr_curves = store_pr_curves


class AnnotationsConfig:
    """
    Configuration for annotation data including class mappings and COCO datasets.
    
    Parameters
    ----------
    names : dict
        Class ID to name mapping
    gt_coco : COCO, optional
        COCO object for ground truths
    predictions_coco : COCO, optional
        COCO object for predictions
    exclude_classes : list, optional
        Class IDs to exclude from evaluation, by default None
    """
    
    def __init__(
        self,
        names: dict,
        gt_coco: COCO = None,
        predictions_coco: COCO = None,
        exclude_classes: list = None
    ):
        self.names = names
        self.gt_coco = gt_coco
        self.predictions_coco = predictions_coco
        self.exclude_classes = exclude_classes or []


class ThresholdsConfig:
    """
    Configuration for detection thresholds.
    
    Parameters
    ----------
    iou_thr : float, optional
        IoU threshold for true positive matching, by default 0.5
    conf_thr : float, optional
        Confidence threshold for predictions, by default 0.5
    """
    
    def __init__(self, iou_thr: float = 0.5, conf_thr: float = 0.5):
        self.iou_thr = iou_thr
        self.conf_thr = conf_thr


class DetectionMetricsState:
    """
    Manages the internal state and data structures for DetectionMetrics.
    
    This class encapsulates all stateful operations and data management,
    providing a clean interface for state updates and resets.
    
    Attributes
    ----------
    matrix : np.ndarray
        Global detection confusion matrix
    multiclass_matrix : np.ndarray
        Multiclass confusion matrix
    class_map : dict
        Mapping from class IDs to matrix indices
    background_idx : int
        Matrix index for background class
    stats : dict
        Accumulated statistics across processed images
    all_preds : list
        All predictions across images (for COCO evaluation)
    all_gts : list
        All ground truths across images
    image_counter : int
        Counter for processed images
    per_class_iou_sum : defaultdict
        Sum of IoU values per class
    per_class_tp_count : defaultdict
        Count of true positives per class
    per_class_union : defaultdict
        Union area accumulator per class
    per_class_intersection : defaultdict
        Intersection area accumulator per class
    pr_gts : list
        Ground truth data for PR curve computation
    pr_preds : list
        Prediction data for PR curve computation
    pr_curves : dict
        Computed Precision-Recall curves
    """
    
    def __init__(self):
        """Initialize state manager with default values."""
        self.reset()
    
    def reset(self) -> None:
        """
        Reset all internal accumulators and state.

        Resets:
        - Confusion matrices
        - Class mapping
        - Statistical accumulators
        - Prediction/ground truth storage
        - Image counter
        - PR curve data
        """
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
        self.pr_gts = []
        self.pr_preds = []
        self.pr_curves = {}
    
    def initialize_global_mapping(self, names: dict, exclude_classes: list[int]) -> None:
        """
        Build class index mapping and initialize confusion matrices.
        
        Parameters
        ----------
        names : dict
            Class ID to name mapping
        exclude_classes : list
            Classes to exclude from evaluation
        """
        valid_ids = [cid for cid in names if cid not in exclude_classes]
        sorted_ids = sorted(valid_ids)
        self.class_map = {cid: idx for idx, cid in enumerate(sorted_ids)}
        self.background_idx = len(self.class_map)

        size = self.background_idx + 1
        self.matrix = np.zeros((size, size), dtype=int)
        self.multiclass_matrix = np.zeros((size, size), dtype=int)
    
    def update_support_count(self, gt_anns: list[Dict]) -> None:
        """
        Increment support count for each class.

        Parameters
        ----------
        gt_anns : list
            List of ground truth annotations
        """
        for ann in gt_anns:
            if ann.get('iscrowd', 0) == 1:
                continue
            cid = ann['category_id']
            if cid in self.class_map:
                self.stats[cid]['support'] += 1
    
    def update_confusion_matrices(self, det_conf: np.ndarray, multi_conf: np.ndarray) -> None:
        """
        Update both confusion matrices with computed values.
        
        Parameters
        ----------
        det_conf : np.ndarray
            Detection confusion matrix for current image
        multi_conf : np.ndarray
            Multiclass confusion matrix for current image
        """
        self.matrix += det_conf
        self.multiclass_matrix += multi_conf
    
    def update_stats_from_confusion(self, confusion: np.ndarray) -> None:
        """
        Update statistics from detection confusion matrix.

        Parameters
        ----------
        confusion : np.ndarray
            Detection confusion matrix
        """
        for cid, idx in self.class_map.items():
            tp = confusion[idx, idx]
            fp = confusion[:, idx].sum() - tp
            fn = confusion[idx, self.background_idx]
            self.stats[cid]['tp'] += int(tp)
            self.stats[cid]['fp'] += int(fp)
            self.stats[cid]['fn'] += int(fn)
    
    def update_iou_metrics(self, cid: int, iou: float, inter: float, gt_area: float, pred_area: float) -> None:
        """
        Update accumulated IoU metrics for a class.

        Parameters
        ----------
        cid : int
            Class ID
        iou : float
            IoU value
        inter : float
            Intersection area
        gt_area : float
            Ground truth area
        pred_area : float
            Predicted area
        """
        union = gt_area + pred_area - inter
        self.per_class_iou_sum[cid] += iou
        self.per_class_tp_count[cid] += 1
        self.per_class_intersection[cid] += inter
        self.per_class_union[cid] += union
    
    def store_predictions_for_coco(self, pred_anns: list[Dict], conf_thr: float) -> None:
        """
        Store predictions above confidence threshold for COCO evaluation.
        
        Parameters
        ----------
        pred_anns : list
            Prediction annotations
        conf_thr : float
            Confidence threshold
        """
        for p in pred_anns:
            if p['score'] >= conf_thr:
                self.all_preds.append({
                    'image_id': self.image_counter,
                    'category_id': p['category_id'],
                    'bbox': p['bbox'],
                    'score': p['score']
                })
    
    def store_pr_data(self, gt_anns: list[Dict], pred_anns: list[Dict]) -> None:
        """
        Store data for Precision-Recall curve computation.
        
        Parameters
        ----------
        gt_anns : list
            Ground truth annotations
        pred_anns : list
            Prediction annotations
        """
        self.pr_gts.append(gt_anns)
        self.pr_preds.append(pred_anns)
    
    def increment_image_counter(self) -> None:
        """Increment the processed image counter."""
        self.image_counter += 1
    
    def get_confusion_matrix_labels(self, names: dict, exclude_classes: list[int]) -> list[str]:
        """
        Get labels for confusion matrix.

        Parameters
        ----------
        names : dict
            Class ID to name mapping
        exclude_classes : list
            Classes to exclude

        Returns
        -------
        list
            Class names with 'background' as last element
        """
        valid_ids = [cid for cid in names if cid not in exclude_classes]
        sorted_ids = sorted(valid_ids)
        labels = [names[cid] for cid in sorted_ids]
        labels.append('background')
        return labels


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

    Parameters
    ----------
    annotations_config : AnnotationsConfig
        Configuration for annotation data
    thresholds_config : ThresholdsConfig
        Configuration for detection thresholds
    pr_config : PrecisionRecallConfig
        Configuration for Precision-Recall data
    """

    def __init__(
        self,
        annotations_config: AnnotationsConfig,
        thresholds_config: ThresholdsConfig = ThresholdsConfig(),
        pr_config: PrecisionRecallConfig =  PrecisionRecallConfig()
    ):
        # Configuration objects
        self.annotations_config = annotations_config
        self.thresholds_config = thresholds_config
        self.pr_config = pr_config
        
        # Initialize state manager
        self.state = DetectionMetricsState()
        self.reset()

    def reset(self) -> None:
        """Reset all internal accumulators and state."""
        self.state.reset()
    
    @property
    def matrix(self) -> List[List[int]]:
        return self.state.matrix
    
    @property
    def pr_curves(self) -> List:
        return self.state.pr_curves
    
    @property
    def multiclass_matrix(self) -> List:
        return self.state.multiclass_matrix

    @property
    def class_map(self):
        return self.state.class_map
    
    def process_image(self, gt_anns: list[Dict], pred_anns: list[Dict]) -> None:
        """
        Process detections for a single image.

        Steps:
        1. Filter excluded classes
        2. Initialize global mapping if needed
        3. Update class support counts
        4. Compute and accumulate confusion matrices
        5. Store predictions for COCO mAP

        Parameters
        ----------
        gt_anns : list
            Ground truth annotations for the image
        pred_anns : list
            Predicted annotations for the image

        Notes
        -----
        - Handles both crowd and non-crowd annotations
        - Stores predictions above confidence threshold for COCO evaluation
        """
        # Filter unwanted classes
        exclude_classes = self.annotations_config.exclude_classes
        gt = [g for g in gt_anns if g['category_id'] not in exclude_classes]
        pr = [p for p in pred_anns if p['category_id'] not in exclude_classes]

        # Initialize global mapping if needed
        if self.state.matrix is None:
            self.state.initialize_global_mapping(
                self.annotations_config.names, 
                exclude_classes
            )

        # Update support counts
        self.state.update_support_count(gt)

        # Calculate IoU matrix
        iou_mat_full = compute_iou_matrix(gt, pr)

        # Compute confusion matrices
        det_conf = self._compute_detection_confusion(gt, pr, iou_mat_full)
        multi_conf = self._compute_multiclass_confusion(gt, pr, iou_mat_full)
        
        # Update state with computed matrices
        self.state.update_confusion_matrices(det_conf, multi_conf)
        self.state.update_stats_from_confusion(det_conf)

        # Store predictions for COCO mAP
        self.state.store_predictions_for_coco(pr, self.thresholds_config.conf_thr)

        # Store data for PR curve computation if enabled
        if self.pr_config.store_pr_data:
            self.state.store_pr_data(gt, pr)
        
        # Increment image counter
        self.state.increment_image_counter()

    def _filter_annotations(
        self,
        gt_anns: list[Dict],
        pred_anns: list[Dict]
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Separate crowd/non-crowd GTs and filter predictions by confidence.

        Parameters
        ----------
        gt_anns : list
            Ground truth annotations
        pred_anns : list
            Prediction annotations

        Returns
        -------
        Tuple[List[dict], List[dict], List[dict]]
            Non-crowd ground truths, crowd ground truths, filtered predictions
        """
        gt_non_crowd = [g for g in gt_anns if g.get('iscrowd', 0) == 0]
        gt_crowd = [g for g in gt_anns if g.get('iscrowd', 0) == 1]
        preds = [p for p in pred_anns if p['score'] >= self.thresholds_config.conf_thr]
        return gt_non_crowd, gt_crowd, preds

    def _handle_detection_conf_edge_cases(
        self,
        gt_non_crowd: list[Dict],
        gt_crowd: list[Dict],
        preds: list[Dict],
        confusion: np.ndarray
    ) -> bool:
        """
        Handle edge cases for detection confusion matrix.

        Parameters
        ----------
        gt_non_crowd : list
            Non-crowd ground truths
        gt_crowd : list
            Crowd ground truths
        preds : list
            Filtered predictions
        confusion : np.ndarray
            Confusion matrix

        Returns
        -------
        bool
            True if edge case handled, False otherwise
        """
        # Case 1: No predictions → all non-crowd GTs are FN
        if not preds:
            for gt in gt_non_crowd:
                cid = gt['category_id']
                if cid in self.state.class_map:
                    idx = self.state.class_map[cid]
                    confusion[idx, self.state.background_idx] += 1
            return True

        # Case 2: No ground truths → all preds are FP
        if not gt_non_crowd and not gt_crowd:
            for pr in preds:
                cid = pr['category_id']
                if cid in self.state.class_map:
                    idx = self.state.class_map[cid]
                    confusion[self.state.background_idx, idx] += 1
            return True

        return False

    def _perform_per_class_matching(
        self,
        gt_non_crowd: list[Dict],
        preds: list[Dict],
        iou_mat: np.ndarray,
        confusion: np.ndarray
    ) -> Tuple[List[bool], List[bool]]:
        """
        Match predictions to ground truths per class using IoU threshold.

        Parameters
        ----------
        gt_non_crowd : list
            Non-crowd ground truths
        preds : list
            Filtered predictions
        iou_mat : np.ndarray
            Precomputed IoU matrix
        confusion : np.ndarray
            Confusion matrix

        Returns
        -------
        Tuple[List[bool], List[bool]]
            Matched status for ground truths and predictions
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
                if best_iou >= self.thresholds_config.iou_thr:
                    gt_matched[best_gt_index] = True
                    pred_matched[j] = True
                    idx = self.state.class_map[cid]
                    confusion[idx, idx] += 1
                    unmatched_gt_list.pop(best_idx)

                    # Calculate areas and intersection
                    gt_box = gt_non_crowd[best_gt_index]['bbox']
                    pred_box = preds[j]['bbox']
                    gt_area = gt_box[2] * gt_box[3]
                    pred_area = pred_box[2] * pred_box[3]
                    inter = best_iou * (gt_area + pred_area) / (1 + best_iou) if best_iou > 0 else 0.0

                    # Update IoU metrics through state manager
                    self.state.update_iou_metrics(cid, best_iou, inter, gt_area, pred_area)
        return gt_matched, pred_matched

    def _process_crowd_matches(
        self,
        gt_crowd: list[Dict],
        preds: list[Dict],
        pred_matched: List[bool]
    ) -> List[bool]:
        """
        Identify predictions matching crowd annotations.

        Parameters
        ----------
        gt_crowd : list
            Crowd ground truths
        preds : list
            Filtered predictions
        pred_matched : List[bool]
            Prediction matched status

        Returns
        -------
        List[bool]
            Updated list indicating crowd-matched predictions
        """
        crowd_matched = [False] * len(preds)
        for crowd in gt_crowd:
            for j, pred in enumerate(preds):
                if pred_matched[j] or crowd_matched[j]:
                    continue
                if crowd['category_id'] != pred['category_id']:
                    continue
                if bbox_iou(crowd['bbox'], pred['bbox']) >= self.thresholds_config.iou_thr:
                    crowd_matched[j] = True
        return crowd_matched

    def _update_confusion_unmatched(
        self,
        gt_non_crowd: list[Dict],
        preds: list[Dict],
        gt_matched: List[bool],
        pred_matched: List[bool],
        crowd_matched: List[bool],
        confusion: np.ndarray
    ) -> None:
        """
        Update confusion matrix for unmatched items.

        Parameters
        ----------
        gt_non_crowd : list
            Non-crowd ground truths
        preds : list
            Filtered predictions
        gt_matched : List[bool]
            GT matched status
        pred_matched : List[bool]
            Prediction matched status
        crowd_matched : List[bool]
            Crowd-matched status
        confusion : np.ndarray
            Confusion matrix
        """
        # Process false negatives (unmatched non-crowd GTs)
        for i, matched in enumerate(gt_matched):
            if not matched:
                cid = gt_non_crowd[i]['category_id']
                if cid in self.state.class_map:
                    idx = self.state.class_map[cid]
                    confusion[idx, self.state.background_idx] += 1

        # Process false positives (unmatched preds not in crowd)
        for j, matched in enumerate(pred_matched):
            if not matched and not crowd_matched[j]:
                cid = preds[j]['category_id']
                if cid in self.state.class_map:
                    idx = self.state.class_map[cid]
                    confusion[self.state.background_idx, idx] += 1

    def _compute_detection_confusion(
        self,
        gt_anns: list[Dict],
        pred_anns: list[Dict],
        iou_mat_full: np.ndarray = None
    ) -> np.ndarray:
        """
        Compute detection confusion matrix (TP, FP, FN).

        Parameters
        ----------
        gt_anns : list
            Ground truth annotations
        pred_anns : list
            Predicted annotations
        iou_mat_full : np.ndarray, optional
            Precomputed IoU matrix, by default None

        Returns
        -------
        np.ndarray
            Detection confusion matrix

        Notes
        -----
        - Handles crowd annotations separately
        - Uses per-class matching strategy
        - Updates IoU metrics for matched pairs
        """
        # Initialize matrix
        size = self.state.background_idx + 1
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
        gt_anns: list[Dict],
        pred_anns: list[Dict],
        iou_matrix: np.ndarray,
        iou_thr: float,
        confusion: np.ndarray
    ) -> tuple[list, bool, np.ndarray]:
        """
        Perform global greedy matching for multiclass confusion matrix.

        Parameters
        ----------
        gt_anns : list
            Ground truth annotations
        pred_anns : list
            Prediction annotations
        iou_matrix : np.ndarray
            IoU matrix between GTs and predictions
        iou_thr : float
            IoU threshold for matching
        confusion : np.ndarray
            Multiclass confusion matrix

        Returns
        -------
        tuple[list, bool, np.ndarray]
            Matched status for GTs, matched status for predictions, updated confusion matrix
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
                if gcls in self.state.class_map and pcls in self.state.class_map:
                    confusion[self.state.class_map[gcls], self.state.class_map[pcls]] += 1

        return gt_matched.tolist(), pred_matched.tolist(), confusion

    def _compute_multiclass_confusion(self,
        gt_anns: list[Dict],
        pred_anns: list[Dict],
        iou_mat_full: np.ndarray = None) -> np.ndarray:
        """
        Compute multiclass confusion matrix.

        Parameters
        ----------
        gt_anns : list
            Ground truth annotations
        pred_anns : list
            Prediction annotations
        iou_mat_full : np.ndarray, optional
            Precomputed IoU matrix, by default None

        Returns
        -------
        np.ndarray
            Multiclass confusion matrix

        Notes
        -----
        - Uses global greedy matching strategy
        - Includes background class for unmatched items
        """
        preds = [p for p in pred_anns if p['score'] >= self.thresholds_config.conf_thr]
        size = self.state.background_idx + 1
        confusion = np.zeros((size, size), dtype=int)
        if self._handle_detection_edge_cases(gt_anns, preds, confusion):
            return confusion
        iou_mat = iou_mat_full

        gt_mat, pred_mat, confusion = self.match_detection_global(
            gt_anns, preds, iou_mat, self.thresholds_config.iou_thr, confusion
        )
        self._handle_unmatched_gts(gt_anns, gt_mat, confusion)
        self._handle_unmatched_preds(preds, pred_mat, confusion)
        return confusion

    def _handle_detection_edge_cases(
        self,
        gt_anns: list[Dict],
        pred_anns: list[Dict],
        confusion: np.ndarray
    ) -> bool:
        """
        Handle edge cases for confusion matrix.

        Parameters
        ----------
        gt_anns : list
            Ground truth annotations
        pred_anns : list
            Prediction annotations
        confusion : np.ndarray
            Confusion matrix

        Returns
        -------
        bool
            True if edge case handled, False otherwise
        """
        if not pred_anns:
            for gt in gt_anns:
                cid = gt['category_id']
                if cid in self.state.class_map:
                    confusion[self.state.class_map[cid], self.state.background_idx] += 1
            return True
        if not gt_anns:
            for pr in pred_anns:
                cid = pr['category_id']
                if cid in self.state.class_map:
                    confusion[self.state.background_idx, self.state.class_map[cid]] += 1
            return True
        return False

    def _handle_unmatched_gts(self,
        gt_anns: list[Dict],
        gt_matched: list[Dict],
        confusion: np.ndarray) -> None:
        """
        Process unmatched ground truths as false negatives.

        Parameters
        ----------
        gt_anns : list
            Ground truth annotations
        gt_matched : list
            GT matched status
        confusion : np.ndarray
            Confusion matrix
        """
        for i, matched in enumerate(gt_matched):
            if not matched:
                cid = gt_anns[i]['category_id']
                if cid in self.state.class_map:
                    confusion[self.state.class_map[cid], self.state.background_idx] += 1

    def _handle_unmatched_preds(self,
        pred_anns: list[Dict],
        pred_matched: list[bool],
        confusion: np.ndarray) -> None:
        """
        Process unmatched predictions as false positives.

        Parameters
        ----------
        pred_anns : list
            Prediction annotations
        pred_matched : list
            Prediction matched status
        confusion : np.ndarray
            Confusion matrix
        """
        for j, matched in enumerate(pred_matched):
            if not matched:
                cid = pred_anns[j]['category_id']
                if cid in self.state.class_map:
                    confusion[self.state.background_idx, self.state.class_map[cid]] += 1

    def compute_metrics(self) -> dict:
        """
        Calculate final evaluation metrics.

        Returns
        -------
        dict
            Comprehensive metrics dictionary containing:
            - Per-class metrics (precision, recall, f1, support, tp, fp, fn, iou, agg_iou, ap)
            - Global metrics (precision, recall, f1, support, tp, fp, fn, mAP, mAP50, mAP75, mIoU)
            - PR curves data (if enabled)

        Notes
        -----
        - Computes COCO mAP if COCO objects are available
        - Calculates both average IoU and aggregate IoU per class
        - Includes PR curve data when store_pr_curves is enabled
        """
        def has_coco_gt(): 
            return self.annotations_config.gt_coco is not None
        
        def has_coco_pred(): 
            return self.annotations_config.predictions_coco is not None

        # Initialize metrics dictionary
        metrics = {}
        total_tp = total_fp = total_fn = total_support = 0

        # Reset PR curves storage
        self.state.pr_curves = {}

        # Compute per-class metrics
        for cid, idx in self.state.class_map.items():
            tp = self.state.stats[cid]['tp']
            fp = self.state.stats[cid]['fp']
            fn = self.state.stats[cid]['fn']
            sup = self.state.stats[cid]['support']
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
            gmap, gmap50, gmap75, per_class_ap = self._compute_map(
                self.annotations_config.gt_coco, 
                self.annotations_config.predictions_coco
            )
            metrics['global']['mAP'] = float(gmap)
            metrics['global']['mAP50'] = float(gmap50)
            metrics['global']['mAP75'] = float(gmap75)

            # Add per-class AP to metrics
            for cid in self.state.class_map:
                ap_val = per_class_ap.get(cid, 0.0)
                metrics[cid]['ap'] = float(ap_val)

        # Compute IoU metrics
        class_ious = []
        for cid in self.state.class_map:
            tp_count = self.state.per_class_tp_count.get(cid, 0)
            iou_sum = self.state.per_class_iou_sum.get(cid, 0.0)
            inter = self.state.per_class_intersection.get(cid, 0.0)
            union = self.state.per_class_union.get(cid, 0.0)

            # Calculate average IoU for matched pairs
            avg_iou = iou_sum / tp_count if tp_count > 0 else 0.0

            # Calculate aggregate IoU
            agg_iou = inter / union if union > 0 else 0.0

            # Update metrics for this class
            if cid in metrics:
                metrics[cid]['iou'] = float(avg_iou)
                metrics[cid]['agg_iou'] = float(agg_iou)
            class_ious.append(agg_iou)

        # Calculate mIoU (mean of aggregate IoU per class)
        mIoU = sum(class_ious) / len(class_ious) if class_ious else 0.0
        metrics['global']['mIoU'] = float(mIoU)

        # Compute PR curves if enabled
        if self.pr_config.store_pr_curves:
            self._compute_pr_curves(metrics)
            metrics['pr_curves'] = self.state.pr_curves

        return metrics

    def _compute_pr_curves(self, metrics: Dict) -> None:
        """
        Compute and store Precision-Recall curves.

        Parameters
        ----------
        metrics : dict
            Computed metrics dictionary
        """
        if not self.annotations_config.gt_coco or not self.annotations_config.predictions_coco:
            return

        # Reset PR curves storage
        self.state.pr_curves = {}
        
        try:
            with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                # Filter out excluded classes
                cat_ids = self.annotations_config.gt_coco.getCatIds()
                valid_cat_ids = [cat_id for cat_id in cat_ids 
                                 if cat_id not in self.annotations_config.exclude_classes]

                evaluator = COCOeval(
                    self.annotations_config.gt_coco, 
                    self.annotations_config.predictions_coco, 
                    'bbox'
                )
                evaluator.params.catIds = valid_cat_ids
                evaluator.evaluate()
                evaluator.accumulate()

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

                self.state.pr_curves['global'] = {
                    'recall': recall_thrs,
                    'precision': np.array(precision_global),
                    'ap': ap_global
                }

                # Per-class PR curves
                for k, cat_id in enumerate(valid_cat_ids):
                    precision_vals = evaluator.eval['precision'][0, :, k, aind, mind]
                    valid_mask = precision_vals > -1

                    self.state.pr_curves[int(cat_id)] = {
                        'recall': recall_thrs[valid_mask],
                        'precision': precision_vals[valid_mask],
                        'ap': metrics.get(int(cat_id), {}).get('ap', 0)
                    }

        except (IndexError, AttributeError, KeyError) as e:
            print(f"Error generating PR curves: {str(e)}")

    def _compute_map(self, gt_coco: COCO, predictions_coco: COCO) -> tuple:
        """
        Compute COCO-style mean Average Precision (mAP).

        Parameters
        ----------
        gt_coco : COCO
            COCO object for ground truths
        predictions_coco : COCO
            COCO object for predictions

        Returns
        -------
        tuple
            (mAP, mAP50, mAP75, per_class_ap)

        Notes
        -----
        - Uses pycocotools for COCO evaluation
        - Computes mAP at IoU thresholds .50:.05:.95
        - Returns per-class AP when available
        """
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            evaluator = COCOeval(gt_coco, predictions_coco, 'bbox')
            cat_ids = gt_coco.getCatIds()
            valid_cat_ids = [cat_id for cat_id in cat_ids 
                             if cat_id not in self.annotations_config.exclude_classes]
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

        Returns
        -------
        float
            Global F1 score
        """
        return self.compute_metrics()['global']['f1']

    @property
    def results_dict(self) -> dict:
        """
        Flattened results dictionary with human-readable keys.

        Returns
        -------
        dict
            Flat mapping of metric names to values with keys formatted as:
            - 'metric/global' for global metrics
            - 'metric/class_name' for per-class metrics
            - 'multiclass_confusion_matrix' for confusion matrix

        Notes
        -----
        - Includes multiclass confusion matrix in the dictionary
        - Uses class names from the names dictionary
        """
        metrics = self.compute_metrics()
        res = {}
        for cid, vals in metrics.items():
            if cid == 'global':
                for k, v in vals.items():
                    res[f"{k}/global"] = v
            else:
                cls_name = self.annotations_config.names.get(cid, f'class_{cid}')
                for k, v in vals.items():
                    res[f"{k}/{cls_name}"] = v
        # Attach multiclass confusion matrix
        res['multiclass_confusion_matrix'] = self.state.multiclass_matrix.tolist()
        return res

    def get_confusion_matrix_labels(self) -> list:
        """
        Get labels for confusion matrix.

        Returns
        -------
        list
            Class names with 'background' as last element

        Notes
        -----
        - Order corresponds to confusion matrix rows/columns
        - Excludes classes specified in exclude_classes
        """
        return self.state.get_confusion_matrix_labels(
            self.annotations_config.names, 
            self.annotations_config.exclude_classes
        )