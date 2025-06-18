"""
    @Author: Matheus Levy
    @Organization: Viplab - UFMA
    @GitHub: https://github.com/viplabufma/MatheusLevy_mestrado
"""
# Object Detection Metrics
import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import contextlib
import io

class DetectionMetrics:
    """
    Computes object detection metrics including precision, recall, F1-score, confusion matrix and mAP.
    
    Attributes:
        names (Dict[int, str]): Dictionary mapping class IDs to class names
        iou_thr (float): IoU threshold for true positive matching (default=0.5)
        conf_thr (float): Confidence threshold for predictions (default=0.5)
        matrix (np.ndarray): Global confusion matrix
        class_map (dict): Mapping from class IDs to matrix indices
        background_idx (int): Index for background class in confusion matrix
        stats (dict): Accumulated statistics across multiple images
        all_preds (list): Stores all predictions across images
        all_gts (list): Stores all ground truths across images
        image_counter (int): Counter for processed images
    
    Methods:
        reset(): Resets the internal state of the metrics calculator
        process_image(gt_anns, pred_anns): Processes detections for a single image
        compute_metrics(): Computes final metrics from accumulated statistics
        _compute_map(): Computes mAP metrics
        ap_per_class(): Computes AP per class (adapted from YOLO)
        compute_ap(): Computes average precision
        fitness(): Returns a single fitness score weighted across metrics
        results_dict: Property returning comprehensive metrics as a dictionary
    """
    
    def __init__(self, names, iou_thr=0.5, conf_thr=0.5, gt_coco=None, predictions_coco=None, exclude_classes=None):
        self.names = names
        self.iou_thr = iou_thr
        self.conf_thr = conf_thr
        self.gt_coco = gt_coco
        self.predictions_coco = predictions_coco
        self.exclude_classes = exclude_classes if exclude_classes is not None else []
        self.reset()
    
    def reset(self):
        """Resets all accumulated statistics and state"""
        self.matrix = None
        self.class_map = {}
        self.background_idx = None
        self.stats = defaultdict(lambda: defaultdict(int))
        self.all_preds = []
        self.all_gts = []
        self.image_counter = 0
        
    def _initialize_global_mapping(self) -> None:
        """Inicializa mapeamento global excluindo classes indesejadas"""
        valid_classes = [cls_id for cls_id in self.names.keys() 
                        if cls_id not in self.exclude_classes]
        
        all_classes = sorted(valid_classes)
        self.class_map = {cls_id: idx for idx, cls_id in enumerate(all_classes)}
        self.background_idx = len(self.class_map)
        
        matrix_size = len(self.class_map) + 1
        self.matrix = np.zeros((matrix_size, matrix_size), dtype=int)
    
    def process_image(self, gt_anns: list, pred_anns: list) -> None:
        # Filtra classes excluídas
        gt_anns = [gt for gt in gt_anns if gt['category_id'] not in self.exclude_classes]
        pred_anns = [pred for pred in pred_anns if pred['category_id'] not in self.exclude_classes]
        
        # Restante do código permanece igual...
        if self.matrix is None:
            self._initialize_global_mapping()
        
        confusion = self._compute_confusion(gt_anns, pred_anns)
        self.matrix += confusion
        self._update_stats_from_confusion(confusion)
        
        # Store predictions (filtrado por classe e confiança)
        for pred in pred_anns:
            if pred['score'] >= self.conf_thr:
                self.all_preds.append({
                    'image_id': self.image_counter,
                    'category_id': pred['category_id'],
                    'bbox': pred['bbox'],
                    'score': pred['score']
                })
        
        self.image_counter += 1
    
    def _compute_confusion(self, gt_anns: list, pred_anns: list) -> np.ndarray:
        """
        Compute confusion matrix for a single image using global class mapping.
        
        Args:
            gt_anns (list): Ground truth annotations
            pred_anns (list): Prediction annotations
            
        Returns:
            confusion (np.ndarray): Confusion matrix
        """
        # Filter predictions by confidence
        pred_anns = [p for p in pred_anns if p['score'] >= self.conf_thr]
        
        # Initialize confusion matrix with global mapping size
        matrix_size = len(self.class_map) + 1
        confusion = np.zeros((matrix_size, matrix_size), dtype=int)
        
        # Handle edge cases
        if self._handle_edge_cases(gt_anns, pred_anns, confusion):
            return confusion
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(gt_anns), len(pred_anns)))
        for i, gt in enumerate(gt_anns):
            for j, pred in enumerate(pred_anns):
                iou_matrix[i, j] = self.bbox_iou(gt['bbox'], pred['bbox'])
        
        # Match detections
        gt_matched, pred_matched = self._match_detections(
            gt_anns, pred_anns, iou_matrix, self.iou_thr, confusion
        )
        
        # Handle unmatched items
        self._handle_unmatched_gts(gt_anns, gt_matched, confusion)
        self._handle_unmatched_preds(pred_anns, pred_matched, confusion)
        
        return confusion

    @staticmethod
    def bbox_iou(box1: list, box2: list) -> float:
        """
        Calculate Intersection over Union (IoU) for COCO-format bounding boxes.
        
        Args:
            box1: [x, y, width, height]
            box2: [x, y, width, height]
            
        Returns:
            iou (float): IoU value (0.0-1.0)
        """
        # Convert to corner coordinates
        x1, y1, w1, h1 = box1
        box1_x1, box1_y1 = x1, y1
        box1_x2, box1_y2 = x1 + w1, y1 + h1
        
        x2, y2, w2, h2 = box2
        box2_x1, box2_y1 = x2, y2
        box2_x2, box2_y2 = x2 + w2, y2 + h2
        
        # Calculate intersection
        inter_x1 = max(box1_x1, box2_x1)
        inter_y1 = max(box1_y1, box2_y1)
        inter_x2 = min(box1_x2, box2_x2)
        inter_y2 = min(box1_y2, box2_y2)
        
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

    def _handle_edge_cases(self, gt_anns: list, pred_anns: list, confusion: np.ndarray) -> bool:
        """Handle cases with no predictions or no ground truths"""
        if not pred_anns:
            for gt in gt_anns:
                cls_id = gt['category_id']
                if cls_id in self.class_map:
                    gt_idx = self.class_map[cls_id]
                    confusion[gt_idx, self.background_idx] += 1  # FN
            return True
        
        if not gt_anns:
            for pred in pred_anns:
                cls_id = pred['category_id']
                if cls_id in self.class_map:
                    pred_idx = self.class_map[cls_id]
                    confusion[self.background_idx, pred_idx] += 1  # FP
            return True
        
        return False

    def _match_detections(self, gt_anns: list, pred_anns, iou_matrix: np.array, iou_thr: float, confusion: np.array) -> tuple[list, list]:
        """Perform two-stage matching (class-specific then class-agnostic)"""
        gt_matched = [False] * len(gt_anns)
        pred_matched = [False] * len(pred_anns)
        
        self._match_optimal_by_class(
            gt_anns, pred_anns, iou_matrix, iou_thr, 
            confusion, gt_matched, pred_matched, 
            same_class=True
        )
        
        self._match_optimal_by_class(
            gt_anns, pred_anns, iou_matrix, iou_thr, 
            confusion, gt_matched, pred_matched, 
            same_class=False
        )
        
        return gt_matched, pred_matched

    def _match_optimal_by_class(self, gt_anns: list, pred_anns: list, iou_matrix: np.array, iou_thr: float, 
                               confusion: np.array, gt_matched: list, pred_matched: list, 
                               same_class=True) -> None:
        """Match detections using Hungarian algorithm with class grouping"""
        class_groups = defaultdict(lambda: {'gt_indices': [], 'pred_indices': []})
        
        # Group unmatched GTs by class
        for i, gt in enumerate(gt_anns):
            if not gt_matched[i]:
                class_groups[gt['category_id']]['gt_indices'].append(i)
        
        # Group unmatched predictions
        for j, pred in enumerate(pred_anns):
            if not pred_matched[j]:
                if same_class:
                    cls_id = pred['category_id']
                    class_groups[cls_id]['pred_indices'].append(j)
                else:
                    for cls_id in class_groups.keys():
                        if cls_id != 'all':  # Don't add to existing class groups
                            continue
                    class_groups['all']['pred_indices'].append(j)
        
        # Process each group
        for cls_id, group in class_groups.items():
            gt_indices = group['gt_indices']
            pred_indices = group['pred_indices']
            
            if not gt_indices or not pred_indices:
                continue
            
            # Create IoU submatrix
            sub_iou = iou_matrix[np.ix_(gt_indices, pred_indices)]
            cost_matrix = 1 - sub_iou  # Convert to minimization problem
            cost_matrix[sub_iou < iou_thr] = 1e9  # Penalize below threshold
            
            # Solve with Hungarian algorithm
            gt_idx, pred_idx = linear_sum_assignment(cost_matrix)
            
            # Process valid matches
            for i, j in zip(gt_idx, pred_idx):
                if cost_matrix[i, j] < 1e8:  # Valid match
                    global_gt_idx = gt_indices[i]
                    global_pred_idx = pred_indices[j]
                    
                    if not gt_matched[global_gt_idx] and not pred_matched[global_pred_idx]:
                        # Mark as matched
                        gt_matched[global_gt_idx] = True
                        pred_matched[global_pred_idx] = True
                        
                        # Get class information
                        gt_cls = gt_anns[global_gt_idx]['category_id']
                        pred_cls = pred_anns[global_pred_idx]['category_id']
                        
                        # Only record if classes are in the global mapping
                        if gt_cls in self.class_map and pred_cls in self.class_map:
                            gt_idx_cls = self.class_map[gt_cls]
                            pred_idx_cls = self.class_map[pred_cls]
                            confusion[gt_idx_cls, pred_idx_cls] += 1

    def _handle_unmatched_gts(self, gt_anns: list, gt_matched: list, confusion: np.array) -> None:
        """Process unmatched ground truths as false negatives"""
        for i, matched in enumerate(gt_matched):
            if not matched:
                cls_id = gt_anns[i]['category_id']
                if cls_id in self.class_map:
                    gt_idx = self.class_map[cls_id]
                    confusion[gt_idx, self.background_idx] += 1  # FN

    def _handle_unmatched_preds(self, pred_anns: list, pred_matched: list, confusion: np.array) -> None:
        """Process unmatched predictions as false positives"""
        for j, matched in enumerate(pred_matched):
            if not matched:
                cls_id = pred_anns[j]['category_id']
                if cls_id in self.class_map:
                    pred_idx = self.class_map[cls_id]
                    confusion[self.background_idx, pred_idx] += 1  # FP


    def _update_stats_from_confusion(self, confusion: np.array) -> None:
        """Update accumulated statistics from confusion matrix"""
        for cls_id, idx in self.class_map.items():
            tp = confusion[idx, idx]
            fp = confusion[self.background_idx, idx]
            fn = confusion[idx, self.background_idx]
            
            # Accumulate statistics
            self.stats[cls_id]['tp'] += tp
            self.stats[cls_id]['fp'] += fp
            self.stats[cls_id]['fn'] += fn
            self.stats[cls_id]['support'] += (tp + fn)
    
    @staticmethod
    def precision(tp: int, fp: int) -> float:
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    @staticmethod
    def recall(tp: int, fn: int) -> float:
        return  tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    @staticmethod
    def f1(precision: float, recall: float) -> float:
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    @staticmethod
    def precision_recall_f1(tp: int, fp: int, fn: int) -> float:
        precision = DetectionMetrics.precision(tp=tp, fp=fp)
        recall = DetectionMetrics.recall(tp=tp, fn=fn)
        return precision, recall, DetectionMetrics.f1(precision=precision, recall=recall)
    
    def compute_metrics(self) -> dict:
        """
        Compute final evaluation metrics including mAP.
        
        Returns:
            metrics (dict): Dictionary containing precision, recall, F1, and mAP metrics
        """
        def TpFpFnSupport(cls_id):
            return self.stats[cls_id]['tp'], self.stats[cls_id]['fp'], self.stats[cls_id]['fn'], self.stats[cls_id]['support']
        
        def hasCOCOgroundtruth() -> bool:
            return self.gt_coco is not None
        def hasCOCOpredictions() -> bool:
            return self.predictions_coco is not None
        
        def update_metrics_with_map(metrics,  global_map, global_map50, global_map75):
            metrics['global']['mAP50'] = global_map50
            metrics['global']['mAP75'] = global_map75
            metrics['global']['mAP'] = global_map     
            return metrics
        
        def compute_mAP_If_Possible(metrics):
            if (hasCOCOgroundtruth() and hasCOCOpredictions()):
                global_map, global_map50, global_map75 = self._compute_map( self.gt_coco, self.predictions_coco)
                metrics = update_metrics_with_map(metrics, global_map, global_map50, global_map75)
            return metrics
        
        def update_total_tp_fp_fn_support(total_tp, total_fp, total_fn, total_support, tp, fp, fn, support):
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_support += support
            return total_tp, total_fp, total_fn, total_support
        
        metrics = {}
        total_tp, total_fp, total_fn, total_support = 0, 0, 0, 0
        
        # Compute per-class mtotal_tpetrics and accumulate totals
        for cls_id in self.class_map.keys():
            tp, fp, fn, support = TpFpFnSupport(cls_id)
            precision, recall, f1 = DetectionMetrics.precision_recall_f1(tp=tp, fp=fp, fn=fn)
            metrics[cls_id] = { 'precision': precision, 'recall': recall, 'f1': f1, 'support': support }
            total_tp, total_fp, total_fn, total_support = update_total_tp_fp_fn_support(total_tp, total_fp, total_fn, total_support, tp, fp, fn, support)

        global_precision, global_recall, global_f1 = DetectionMetrics.precision_recall_f1(tp=total_tp, fp=total_fp, fn=total_fn)
        metrics['global'] = { 'precision': global_precision, 'recall': global_recall, 'f1': global_f1, 'support': total_support }
        metrics = compute_mAP_If_Possible(metrics)
        return metrics

    def _compute_map(self, gt_coco: COCO, predictions_coco: COCO) -> dict:
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            coco_eval = COCOeval(gt_coco, predictions_coco, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        return coco_eval.stats[0], coco_eval.stats[1], coco_eval.stats[2]
    
    @property
    def fitness(self) -> float:
        """
        Compute a single fitness score based on precision, recall, and F1.
        
        Returns:
            fitness (float): Weighted average of precision, recall, and F1
        """
        metrics = self.compute_metrics()
        global_metrics = metrics['global']
        
        # Use F1 as the primary metric for fitness
        return global_metrics['f1']
    
    @property
    def results_dict(self) -> dict:
        """Comprehensive results including global metrics and mAP"""
        metrics = self.compute_metrics()
        results = {}
        
        # Per-class metrics
        for cls_id, cls_metrics in metrics.items():
            if cls_id == 'global':
                # Handle global metrics separately
                for k, v in cls_metrics.items():
                    results[f'{k}/global'] = v
            else:
                cls_name = self.names.get(cls_id, f'class_{cls_id}')
                for metric, value in cls_metrics.items():
                    results[f'{metric}/{cls_name}'] = value
        
        return results