"""
    @Author: Matheus Levy
    @Organization: Viplab - UFMA
    @GitHub: https://github.com/viplabufma/MatheusLevy_mestrado
"""
import json
import tempfile
import contextlib
import io

import numpy as np
import pytest
from detmet import DetectionMetrics, bbox_iou
from detmet.metrics import (
    compute_iou_matrix,
    compute_iou_matrix_xywh,
    compute_iou_matrix_xyxy,
    compute_precision_recall_curve,
)
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def _build_coco_batches(all_gts, all_preds):
    """Build COCO GT/prediction payloads from list-of-images annotations."""
    n_images = max(len(all_gts), len(all_preds))
    all_gts_padded = list(all_gts) + [[] for _ in range(max(0, n_images - len(all_gts)))]
    all_preds_padded = list(all_preds) + [[] for _ in range(max(0, n_images - len(all_preds)))]

    cat_ids = sorted(
        {
            int(ann["category_id"])
            for image_anns in (all_gts_padded + all_preds_padded)
            for ann in image_anns
        }
    )

    coco_gt = {
        "info": {"description": "Runtime test fixture", "version": "1.0", "year": 2026},
        "licenses": [],
        "images": [
            {"id": i + 1, "file_name": f"img_{i + 1}.jpg", "width": 640, "height": 640}
            for i in range(n_images)
        ],
        "annotations": [],
        "categories": [
            {"id": cat_id, "name": f"class_{cat_id}", "supercategory": "thing"}
            for cat_id in cat_ids
        ],
    }

    prediction_rows = []
    ann_id = 1

    for idx, (img_gts, img_preds) in enumerate(zip(all_gts_padded, all_preds_padded), start=1):
        for gt in img_gts:
            x, y, w, h = gt["bbox"]
            coco_gt["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": idx,
                    "category_id": int(gt["category_id"]),
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "area": float(w) * float(h),
                    "iscrowd": int(gt.get("iscrowd", 0)),
                }
            )
            ann_id += 1

        for pred in img_preds:
            x, y, w, h = pred["bbox"]
            prediction_rows.append(
                {
                    "image_id": idx,
                    "category_id": int(pred["category_id"]),
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "score": float(pred.get("score", 1.0)),
                }
            )

    return coco_gt, prediction_rows


def _compute_oracle_from_batches(
    all_gts,
    all_preds,
    iou_threshold=0.5,
    class_agnostic=False,
):
    """Compute PR-AP oracle matching compute_precision_recall_curve semantics."""
    coco_gt, pred_rows = _build_coco_batches(all_gts, all_preds)

    with tempfile.TemporaryDirectory() as tmpdir:
        gt_path = f"{tmpdir}/gt.json"
        pred_path = f"{tmpdir}/pred.json"
        with open(gt_path, "w") as f:
            json.dump(coco_gt, f)
        with open(pred_path, "w") as f:
            json.dump(pred_rows, f)

        gt_coco = COCO(gt_path)
        pred_coco = gt_coco.loadRes(pred_path)

        evaluator = COCOeval(gt_coco, pred_coco, "bbox")
        evaluator.params.iouThrs = np.array([iou_threshold], dtype=float)
        if class_agnostic:
            evaluator.params.useCats = 0

        with contextlib.redirect_stdout(io.StringIO()):
            evaluator.evaluate()
            evaluator.accumulate()

        if not hasattr(evaluator, "eval") or "precision" not in evaluator.eval:
            return 0.0, {}

        precision = evaluator.eval["precision"]  # [T, R, K, A, M]
        rec_thrs = evaluator.params.recThrs

        try:
            aind = next(i for i, label in enumerate(evaluator.params.areaRngLbl) if label == "all")
        except StopIteration:
            aind = 0
        try:
            mind = next(i for i, max_det in enumerate(evaluator.params.maxDets) if max_det == 100)
        except StopIteration:
            mind = len(evaluator.params.maxDets) - 1

        iou_idx = 0
        precision_global = []
        for r_idx in range(len(rec_thrs)):
            vals = []
            for k in range(precision.shape[2]):
                val = precision[iou_idx, r_idx, k, aind, mind]
                if val > -1:
                    vals.append(float(val))
            precision_global.append(float(np.mean(vals)) if vals else 0.0)

        ap_global = float(np.mean(precision_global)) if precision_global else 0.0

        per_class_ap = {}
        cat_ids = list(evaluator.params.catIds) if hasattr(evaluator.params, "catIds") else []
        for idx, cat_id in enumerate(cat_ids):
            pr = precision[iou_idx, :, idx, aind, mind]
            valid_pr = pr[pr > -1]
            ap = float(valid_pr.mean()) if valid_pr.size > 0 else 0.0
            per_class_ap[int(cat_id)] = ap

        # Ensure classes present only in predictions are represented with AP=0.0
        classes_from_inputs = {
            int(ann["category_id"])
            for image_anns in (all_gts + all_preds)
            for ann in image_anns
        }
        for class_id in classes_from_inputs:
            per_class_ap.setdefault(class_id, 0.0)

        return ap_global, per_class_ap

def test_multi_image_processing():
    """
    Test cumulative processing of multiple images with different scenarios.

    Verifies:
    - Correct accumulation of confusion matrix across images
    - Per-class metric calculation after multiple images
    - Handling of FPs when prediction doesn't match any GT (mapped as background)
    - Scenarios: 
        * Image 1: TP class1, TP class2 + FP class3 
        * Image 2: TP class2 + FP class2 (class mismatch) and FP class3
    """
    # Image 1 setup
    gt_anns1 = [{'category_id': 1, 'bbox': [10, 10, 20, 20]}]
    pred_anns1 = [{'category_id': 1, 'bbox': [12, 12, 18, 18], 'score': 0.9}]
    
    # Image 2 setup
    gt_anns2 = [{'category_id': 2, 'bbox': [50, 50, 20, 20]}]
    pred_anns2 = [
        {'category_id': 2, 'bbox': [52, 52, 18, 18], 'score': 0.9},
        {'category_id': 3, 'bbox': [100, 100, 30, 30], 'score': 0.8}
    ]
    
    # Process images sequentially
    metrics = DetectionMetrics(names={1: 'class1', 2: 'class2', 3: 'class3'})
    metrics.process_image(gt_anns1, pred_anns1)
    metrics.process_image(gt_anns2, pred_anns2)
    
    # Validate accumulated confusion matrix
    # Global mapping: {1: 0, 2: 1, 3: 2}, background_idx = 3
    expected_matrix = np.zeros((4, 4), dtype=int)
    expected_matrix[0, 0] = 1  # TP class1 (index 0)
    expected_matrix[1, 1] = 1  # TP class2 (index 1)
    expected_matrix[3, 2] = 1  # FP class3 (prediction index 2 mapped to background index 3)
    
    assert np.array_equal(metrics.matrix, expected_matrix)
    
    # Validate per-class metrics
    metrics_result = metrics.compute_metrics()
    assert metrics_result[1]['precision'] == 1.0
    assert metrics_result[1]['recall'] == 1.0
    assert metrics_result[1]['iou'] == 0.8100000023841858
    assert metrics_result[2]['precision'] == 1.0
    assert metrics_result[2]['recall'] == 1.0
    assert metrics_result[2]['iou'] == 0.8100000023841858
    assert metrics_result[3]['precision'] == 0.0  # 0 TP, 1 FP
    assert metrics_result[3]['recall'] == 0.0     # 0 FN (no GT for class3)
    assert metrics_result[3]['iou'] == 0.0
    assert metrics_result['global']['mIoU'] == 0.5400000214576721

def test_single_image_basic():
    """
    Basic test with single image and perfect detections.

    Verifies:
    - Correct matching when IoU > threshold
    - Precision/recall=1.0 for all classes
    - Processing multiple instances per image
    - IoU calculation accuracy for matched boxes
    """
    gt_anns = [
        {'category_id': 1, 'bbox': [10, 10, 20, 20]},
        {'category_id': 2, 'bbox': [50, 50, 30, 30]}
    ]
    pred_anns = [
        {'category_id': 1, 'bbox': [12, 12, 18, 18], 'score': 0.9},
        {'category_id': 2, 'bbox': [52, 52, 28, 28], 'score': 0.8}
    ]
    
    metrics = DetectionMetrics(names={1: 'person', 2: 'car'})
    metrics.process_image(gt_anns, pred_anns)
    
    result = metrics.compute_metrics()
    
    # All classes should have maximum precision and recall
    assert result[1]['precision'] == 1.0
    assert result[1]['recall'] == 1.0
    assert result[1]['iou'] == 0.8100000023841858
    assert result[2]['precision'] == 1.0
    assert result[2]['recall'] == 1.0
    assert result[2]['iou'] == 0.8711110949516296
    assert result['global']['mIoU'] == 0.8405554890632629

def test_no_predictions():
    """
    Test scenario with no predictions (model fails to detect any objects).

    Verifies:
    - Precision and recall should be 0.0 when there are GTs but no predictions
    - Support count should reflect number of GTs
    - FN should equal number of GTs
    - IoU metrics remain 0.0 with no detections
    """
    gt_anns = [
        {'category_id': 1, 'bbox': [10, 10, 20, 20]},
        {'category_id': 2, 'bbox': [50, 50, 30, 30]}
    ]
    pred_anns = []
    
    metrics = DetectionMetrics(names={1: 'person', 2: 'car'})
    metrics.process_image(gt_anns, pred_anns)
    
    result = metrics.compute_metrics()
    
    # No predictions: precision undefined (treated as 0), recall = 0
    assert result[1]['precision'] == 0.0
    assert result[1]['recall'] == 0.0
    assert result[2]['precision'] == 0.0
    assert result[2]['recall'] == 0.0
    assert result[1]['support'] == 1  # 1 GT present
    assert result[2]['support'] == 1
    assert result['global']['mIoU'] == 0.0

def test_no_ground_truth():
    """Tests scenario with no annotations (predictions should be considered FP).
    
    Verifies:
    - Precision should be 0 when no GTs but predictions exist
    - Recall undefined (treated as 0) when no GTs
    - Support should be 0 for all classes
    """
    gt_anns = []
    pred_anns = [
        {'category_id': 1, 'bbox': [12, 12, 18, 18], 'score': 0.9},
        {'category_id': 2, 'bbox': [52, 52, 28, 28], 'score': 0.8}
    ]
    
    metrics = DetectionMetrics(names={1: 'person', 2: 'car'})
    metrics.process_image(gt_anns, pred_anns)
    
    result = metrics.compute_metrics()
    
    # No GTs: all predictions are FP
    assert result[1]['precision'] == 0.0
    assert result[1]['recall'] == 0.0
    assert result[2]['precision'] == 0.0
    assert result[2]['recall'] == 0.0
    assert result[1]['support'] == 0  # No GT for class
    assert result[2]['support'] == 0
    assert result['global']['mIoU'] == 0.0

def test_confidence_threshold():
    """Tests filtering by confidence threshold.
    
    Verifies:
    - Predictions below confidence threshold are ignored
    - Only detections with sufficient confidence are processed
    - Effect on precision/recall calculation
    """
    gt_anns = [{'category_id': 1, 'bbox': [10, 10, 20, 20]}]
    pred_anns = [
        {'category_id': 1, 'bbox': [12, 12, 18, 18], 'score': 0.9},  # Above threshold
        {'category_id': 1, 'bbox': [15, 15, 15, 15], 'score': 0.3}   # Below threshold
    ]
    
    metrics = DetectionMetrics(names={1: 'person'}, conf_thr=0.5)
    metrics.process_image(gt_anns, pred_anns)
    
    result = metrics.compute_metrics()
    
    # Only 1 valid prediction should be considered (TP)
    assert result[1]['precision'] == 1.0
    assert result[1]['recall'] == 1.0
    assert result['global']['mIoU'] == 0.8100000023841858

def test_iou_threshold():
    """Tests matching based on IoU threshold.
    
    Verifies:
    - Detections with IoU < threshold are not considered TP
    - Low IoU results in FP (prediction) and FN (undetected GT)
    - Impact on precision and recall metrics
    """
    gt_anns = [{'category_id': 1, 'bbox': [10, 10, 20, 20]}]
    pred_anns = [
        {'category_id': 1, 'bbox': [30, 30, 20, 20], 'score': 0.9}  # Low IoU
    ]
    
    metrics = DetectionMetrics(names={1: 'person'}, iou_thr=0.5)
    metrics.process_image(gt_anns, pred_anns)
    
    result = metrics.compute_metrics()
    
    # Low IoU: FP (unmatched prediction) and FN (undetected GT)
    assert result[1]['precision'] == 0.0  # 0 TP, 1 FP
    assert result[1]['recall'] == 0.0     # 0 TP, 1 FN
    assert result['global']['mIoU'] == 0.0

def test_class_mismatch():
    """Tests predictions with incorrect class (classification error).
    
    Verifies:
    - When predicted class ≠ GT class:
        * Original GT counts as FN for its class
        * Prediction counts as FP for predicted class
    - Impact on metrics for both involved classes
    """
    gt_anns = [{'category_id': 1, 'bbox': [10, 10, 20, 20]}]
    pred_anns = [{'category_id': 2, 'bbox': [12, 12, 18, 18], 'score': 0.9}]
    
    metrics = DetectionMetrics(names={1: 'person', 2: 'car'})
    metrics.process_image(gt_anns, pred_anns)
    
    result = metrics.compute_metrics()
    
    # Class 1: FN (GT not correctly detected)
    assert result[1]['precision'] == 0.0
    assert result[1]['recall'] == 0.0
    # Class 2: FP (incorrect prediction assigned to class)
    assert result[2]['precision'] == 0.0
    assert result[2]['recall'] == 0.0
    assert result['global']['mIoU'] == 0.0

def test_multiple_predictions_same_gt():
    """Tests multiple predictions for same GT (duplicate detections).
    
    Verifies:
    - Only 1 prediction is considered TP (highest confidence)
    - Additional predictions for same object count as FP
    - Maximum recall (GT detected) but reduced precision
    """
    gt_anns = [{'category_id': 1, 'bbox': [10, 10, 20, 20]}]
    pred_anns = [
        {'category_id': 1, 'bbox': [12, 12, 18, 18], 'score': 0.9},
        {'category_id': 1, 'bbox': [11, 11, 19, 19], 'score': 0.8}
    ]
    
    metrics = DetectionMetrics(names={1: 'person'})
    metrics.process_image(gt_anns, pred_anns)
    
    result = metrics.compute_metrics()
    
    # 1 TP + 1 FP = precision 0.5, recall 1.0
    assert result[1]['precision'] == 0.5
    assert result[1]['recall'] == 1.0
    assert result['global']['mIoU'] == 0.8100000023841858
 
def test_bbox_iou_calculation():
    """Tests direct IoU calculation between bounding boxes.
    
    Covered cases:
    - Identical boxes (IoU=1.0)
    - Non-overlapping boxes (IoU=0.0)
    - Partially overlapping boxes (0.0 < IoU < 1.0)
    """
  
    # Identical boxes
    iou1 = bbox_iou([10, 10, 20, 20], [10, 10, 20, 20])
    assert iou1 == 1.0
    
    # Disjoint boxes
    iou2 = bbox_iou([10, 10, 20, 20], [50, 50, 20, 20])
    assert iou2 == 0.0
    
    # Partial overlap
    iou3 = bbox_iou([10, 10, 20, 20], [20, 20, 20, 20])
    assert iou3 > 0.0 and iou3 < 1.0


def test_compute_iou_matrix_xywh_matches_bbox_iou():
    """IoU matrix for COCO xywh must match scalar bbox_iou for the same pair."""
    gt_anns = [{'bbox': [10, 10, 20, 20]}]
    pred_anns = [{'bbox': [15, 15, 20, 20]}]

    iou_mat = compute_iou_matrix_xywh(gt_anns, pred_anns)
    expected_iou = bbox_iou(gt_anns[0]['bbox'], pred_anns[0]['bbox'])

    assert iou_mat.shape == (1, 1)
    assert iou_mat[0, 0] == pytest.approx(expected_iou)


def test_compute_iou_matrix_xyxy_uses_direct_xyxy():
    """IoU matrix for xyxy inputs should be computed directly with no conversion."""
    gt_boxes = np.array([[10, 10, 30, 30]], dtype=np.float32)
    pred_boxes = np.array([[15, 15, 35, 35]], dtype=np.float32)

    iou_mat = compute_iou_matrix_xyxy(gt_boxes, pred_boxes)
    expected_iou = 225.0 / 575.0

    assert iou_mat.shape == (1, 1)
    assert iou_mat[0, 0] == pytest.approx(expected_iou)


def test_compute_iou_matrix_xywh_and_xyxy_are_equivalent():
    """Equivalent boxes in xywh/xyxy must produce the same IoU matrix."""
    gt_anns_xywh = [{'bbox': [10, 10, 20, 20]}]
    pred_anns_xywh = [{'bbox': [15, 15, 20, 20]}]

    iou_xywh = compute_iou_matrix_xywh(gt_anns_xywh, pred_anns_xywh)

    gt_boxes_xyxy = np.array([[10, 10, 30, 30]], dtype=np.float32)
    pred_boxes_xyxy = np.array([[15, 15, 35, 35]], dtype=np.float32)
    iou_xyxy = compute_iou_matrix_xyxy(gt_boxes_xyxy, pred_boxes_xyxy)

    assert np.allclose(iou_xywh, iou_xyxy)


def test_compute_iou_matrix_wrapper_delegates_by_format():
    """Backward-compatible wrapper should delegate to the explicit format APIs."""
    gt_anns_xywh = [{'bbox': [10, 10, 20, 20]}]
    pred_anns_xywh = [{'bbox': [15, 15, 20, 20]}]

    wrapped_xywh = compute_iou_matrix(gt_anns_xywh, pred_anns_xywh)
    explicit_xywh = compute_iou_matrix_xywh(gt_anns_xywh, pred_anns_xywh)
    assert np.allclose(wrapped_xywh, explicit_xywh)

    gt_anns_xyxy = [{'bbox': [10, 10, 30, 30]}]
    pred_anns_xyxy = [{'bbox': [15, 15, 35, 35]}]
    wrapped_xyxy = compute_iou_matrix(gt_anns_xyxy, pred_anns_xyxy, bbox_format='xyxy')

    explicit_xyxy = compute_iou_matrix_xyxy(
        np.array([g['bbox'] for g in gt_anns_xyxy], dtype=np.float32),
        np.array([p['bbox'] for p in pred_anns_xyxy], dtype=np.float32),
    )
    assert np.allclose(wrapped_xyxy, explicit_xyxy)

def test_results_dict_property():
    """Tests comprehensive results dictionary with structured keys.
    
    Verifies:
    - Key formatting (prefix/class)
    - Inclusion of all per-class metrics (precision, recall, f1, support)
    - Presence of entries for all configured classes
    """
    gt_anns = [{'category_id': 1, 'bbox': [10, 10, 20, 20]}]
    pred_anns = [{'category_id': 1, 'bbox': [12, 12, 18, 18], 'score': 0.9}]
    
    metrics = DetectionMetrics(names={1: 'person', 2: 'car'})
    metrics.process_image(gt_anns, pred_anns)
    
    results = metrics.results_dict
    
    # Verify expected keys for each class
    expected_keys = [
        'precision/person', 'recall/person', 'f1/person', 'support/person',
        'precision/car', 'recall/car', 'f1/car', 'support/car'
    ]
    
    for key in expected_keys:
        assert key in results

def test_reset_functionality():
    """Tests reset() method for internal state reinitialization.
    
    Verifies:
    - Confusion matrix is reset to None
    - Accumulated statistics are zeroed
    - Class mapping remains configured
    """
    gt_anns = [{'category_id': 1, 'bbox': [10, 10, 20, 20]}]
    pred_anns = [{'category_id': 1, 'bbox': [12, 12, 18, 18], 'score': 0.9}]
    
    metrics = DetectionMetrics(names={1: 'person'})
    metrics.process_image(gt_anns, pred_anns)
    
    # Initial populated state
    assert metrics.matrix is not None
    assert len(metrics.stats) > 0
    
    # Reset
    metrics.reset()
    
    # Post-reset state
    assert metrics.matrix is None
    assert len(metrics.stats) == 0
    assert len(metrics.class_map) == 0  # Mapping should be preserved?

def test_edge_case_empty_inputs():
    """Tests edge case with empty inputs (no GTs and no predictions).
    
    Verifies:
    - Metrics should be 0 when no data
    - Support should be 0
    - Processing shouldn't generate errors
    """
    metrics = DetectionMetrics(names={1: 'person'})
    metrics.process_image([], [])
    
    result = metrics.compute_metrics()
    
    # Empty inputs: undefined metrics (treated as 0)
    assert result[1]['precision'] == 0.0
    assert result[1]['recall'] == 0.0
    assert result[1]['support'] == 0  # No GT present

def test_three_images_metrics():
    """Advanced test with 3 images verifying:
    
    - Complex accumulated confusion matrix
    - Per-class metric calculation (precision, recall, f1, support)
    - Consolidated global metrics
    - Fitness based on global F1
    - Effect of confidence threshold on predictions
    
    Scenarios:
      Img1: 2 TP (class1, class2) + 1 FP (class3)
      Img2: 1 TP (class1) + 1 TP (class3) + 1 FP (class2)
      Img3: 1 TP (class2) + 1 TP (class3) + 1 FP (class1) + ignored prediction (conf < thr)
    """
    # Image 1 setup
    gt_anns1 = [
        {'category_id': 1, 'bbox': [10, 10, 20, 20]},
        {'category_id': 2, 'bbox': [50, 50, 20, 20]}
    ]
    pred_anns1 = [
        {'category_id': 1, 'bbox': [12, 12, 18, 18], 'score': 0.9},  # TP class1
        {'category_id': 2, 'bbox': [52, 52, 18, 18], 'score': 0.9},  # TP class2
        {'category_id': 3, 'bbox': [100, 100, 30, 30], 'score': 0.8} # FP class3
    ]
    
    # Image 2 setup
    gt_anns2 = [
        {'category_id': 1, 'bbox': [10, 10, 20, 20]},
        {'category_id': 3, 'bbox': [50, 50, 20, 20]}
    ]
    pred_anns2 = [
        {'category_id': 1, 'bbox': [12, 12, 18, 18], 'score': 0.9},  # TP class1
        {'category_id': 2, 'bbox': [100, 100, 30, 30], 'score': 0.8}, # FP class2
        {'category_id': 3, 'bbox': [52, 52, 18, 18], 'score': 0.9}   # TP class3
    ]
    
    # Image 3 setup
    gt_anns3 = [
        {'category_id': 2, 'bbox': [10, 10, 20, 20]},
        {'category_id': 3, 'bbox': [50, 50, 20, 20]}
    ]
    pred_anns3 = [
        {'category_id': 1, 'bbox': [100, 100, 30, 30], 'score': 0.8}, # FP class1
        {'category_id': 2, 'bbox': [12, 12, 18, 18], 'score': 0.9},   # TP class2
        {'category_id': 3, 'bbox': [52, 52, 18, 18], 'score': 0.6},   # TP class3 (conf > thr)
        {'category_id': 3, 'bbox': [55, 55, 15, 15], 'score': 0.3}    # Ignored (conf < thr)
    ]
    
    # Process all 3 images
    metrics = DetectionMetrics(names={1: 'class1', 2: 'class2', 3: 'class3'})
    metrics.process_image(gt_anns1, pred_anns1)
    metrics.process_image(gt_anns2, pred_anns2)
    metrics.process_image(gt_anns3, pred_anns3)
    
    # 1. Validate final confusion matrix
    # Mapping: {1:0, 2:1, 3:2}, background_idx=3
    expected_matrix = np.zeros((4, 4), dtype=int)
    expected_matrix[0, 0] = 2  # class1 TP (2 images)
    expected_matrix[1, 1] = 2  # class2 TP (img1 and img3)
    expected_matrix[2, 2] = 2  # class3 TP (img2 and img3)
    # FPs (background -> prediction)
    expected_matrix[3, 0] = 1  # FP class1 (img3)
    expected_matrix[3, 1] = 1  # FP class2 (img2)
    expected_matrix[3, 2] = 1  # FP class3 (img1)
    
    assert np.array_equal(metrics.matrix, expected_matrix), \
        f"Incorrect confusion matrix.\nExpected:\n{expected_matrix}\nGot:\n{metrics.matrix}"
    
    # 2. Per-class metrics
    metrics_result = metrics.compute_metrics()
    
    # Class 1: 2 TP, 1 FP, 0 FN
    assert metrics_result[1]['precision'] == 2/3
    assert metrics_result[1]['recall'] == 1.0
    assert metrics_result[1]['f1'] == (2 * (2/3) * 1.0) / ((2/3) + 1.0)
    assert metrics_result[1]['support'] == 2
    assert metrics_result[1]['iou'] == 0.8100000023841858
    
    # Class 2: 2 TP, 1 FP, 0 FN
    assert metrics_result[2]['precision'] == 2/3
    assert metrics_result[2]['recall'] == 1.0
    assert metrics_result[2]['f1'] == (2 * (2/3) * 1.0) / ((2/3) + 1.0)
    assert metrics_result[2]['support'] == 2
    assert metrics_result[2]['iou'] == 0.8100000023841858
    
    # Class 3: 2 TP, 1 FP, 0 FN
    assert metrics_result[3]['precision'] == 2/3
    assert metrics_result[3]['recall'] == 1.0
    assert metrics_result[3]['f1'] == (2 * (2/3) * 1.0) / ((2/3) + 1.0)
    assert metrics_result[3]['support'] == 2
    assert metrics_result[3]['iou'] == 0.8100000023841858
    
    # 3. Global metrics
    global_metrics = metrics_result['global']
    assert global_metrics['precision'] == 6/(6+3)
    assert global_metrics['recall'] == 1.0
    expected_f1 = 2 * (6/9 * 1.0) / (6/9 + 1.0)
    assert abs(global_metrics['f1'] - expected_f1) < 1e-6
    assert global_metrics['support'] == 6
    assert global_metrics['mIoU'] == 0.8100000023841858

    # 4. Verify results dictionary
    results_dict = metrics.results_dict
    assert results_dict['precision/class1'] == 2/3
    assert results_dict['recall/class1'] == 1.0
    assert results_dict['f1/class1'] == (2 * (2/3) * 1.0) / ((2/3) + 1.0)
    assert results_dict['support/class1'] == 2
    
    assert results_dict['precision/class2'] == 2/3
    assert results_dict['recall/class2'] == 1.0
    assert results_dict['f1/class2'] == (2 * (2/3) * 1.0) / ((2/3) + 1.0)
    assert results_dict['support/class2'] == 2
    
    assert results_dict['precision/class3'] == 2/3
    assert results_dict['recall/class3'] == 1.0
    assert results_dict['f1/class3'] == (2 * (2/3) * 1.0) / ((2/3) + 1.0)
    assert results_dict['support/class3'] == 2
    
    # Global metrics in dictionary
    assert results_dict['precision/global'] == 6/(6+3)
    assert results_dict['recall/global'] == 1.0
    assert abs(results_dict['f1/global'] - expected_f1) < 1e-6
    assert results_dict['support/global'] == 6

    # Verify fitness metric
    assert metrics.fitness == results_dict['f1/global']

def test_map_calculation():
    """Comprehensive test for mAP (mean Average Precision) calculation.
    
    Verifies:
    - Processing multiple classes
    - Effect of different confidence levels
    - mAP calculation with different IoU thresholds
    - Complex scenarios with:
        * TPs, FPs, FNs
        * Duplicate detections
        * Confidence variations
    """
    names = {0: 'person', 1: 'car', 2: 'bike'}
    metrics = DetectionMetrics(names=names, iou_thr=0.5, conf_thr=0.1)
    
    # Image 1: 3 TPs (all classes)
    gt_anns_1 = [
        {'category_id': 0, 'bbox': [10, 10, 20, 20]},  
        {'category_id': 1, 'bbox': [50, 50, 30, 30]},  
        {'category_id': 2, 'bbox': [100, 100, 15, 15]} 
    ]
    pred_anns_1 = [
        {'category_id': 0, 'bbox': [10, 10, 20, 20], 'score': 0.95}, 
        {'category_id': 1, 'bbox': [50, 50, 30, 30], 'score': 0.90}, 
        {'category_id': 2, 'bbox': [100, 100, 15, 15], 'score': 0.85}
    ]
    metrics.process_image(gt_anns_1, pred_anns_1)
    
    # Image 2: 2 TPs + 1 FP + 1 extra detection (class2)
    gt_anns_2 = [
        {'category_id': 0, 'bbox': [200, 200, 25, 25]},
        {'category_id': 1, 'bbox': [300, 300, 40, 40]},
    ]
    pred_anns_2 = [
        {'category_id': 0, 'bbox': [202, 202, 23, 23], 'score': 0.88},  # TP
        {'category_id': 1, 'bbox': [305, 305, 35, 35], 'score': 0.82},  # TP
        {'category_id': 0, 'bbox': [400, 400, 20, 20], 'score': 0.75},  # FP (class0)
        {'category_id': 2, 'bbox': [500, 500, 10, 10], 'score': 0.70}   # FP (class2)
    ]
    metrics.process_image(gt_anns_2, pred_anns_2)
    
    # Image 3: 1 TP (class1) with low confidence (> thr)
    gt_anns_3 = [
        {'category_id': 1, 'bbox': [600, 600, 50, 50]},
    ]
    pred_anns_3 = [
        {'category_id': 1, 'bbox': [620, 620, 30, 30], 'score': 0.60},  # TP (valid IoU)
    ]
    metrics.process_image(gt_anns_3, pred_anns_3)
    
    # Image 4: 1 GT + multiple predictions (1 TP + 2 FPs)
    gt_anns_4 = [
        {'category_id': 0, 'bbox': [700, 700, 30, 30]},
    ]
    pred_anns_4 = [
        {'category_id': 0, 'bbox': [702, 702, 28, 28], 'score': 0.95},  # TP
        {'category_id': 0, 'bbox': [705, 705, 25, 25], 'score': 0.85},  # FP (duplicate)
        {'category_id': 0, 'bbox': [710, 710, 20, 20], 'score': 0.75},  # FP
    ]
    metrics.process_image(gt_anns_4, pred_anns_4)
    result = metrics.compute_metrics()

    # Verify presence of global metrics
    assert 'global' in result, "Global metrics should be present"

    all_gts = [gt_anns_1, gt_anns_2, gt_anns_3, gt_anns_4]
    all_preds = [pred_anns_1, pred_anns_2, pred_anns_3, pred_anns_4]
    # Validate AP behavior via COCO-backed PR computation on the same batches.
    pr_result = compute_precision_recall_curve(all_gts, all_preds, iou_threshold=0.5)
    exp_ap, exp_per_class = _compute_oracle_from_batches(all_gts, all_preds, iou_threshold=0.5)

    assert float(pr_result['ap']) == pytest.approx(exp_ap, rel=1e-6)
    for class_id, expected_ap in exp_per_class.items():
        assert class_id in pr_result['per_class']
        assert float(pr_result['per_class'][class_id]) == pytest.approx(expected_ap, rel=1e-6)

def test_classification_errors_in_confusion_matrix():
    """Tests accounting of classification errors in multiclass matrix.
    
    Scenario:
      - Prediction class2 for GT class1 (error)
      - Prediction class3 with no matching GT (traditional FP)
    
    Verifies:
    - Confusion matrix reflects:
        * GT class1 predicted as class2 (FN class1 + FP class2)
        * FP class3
    - Per-class metrics reflect classification error:
        * Class1: FN
        * Class2: TP + FP (from error)
        * Class3: Traditional FP
    """
    names = {1: 'class1', 2: 'class2', 3: 'class3'}
    metrics = DetectionMetrics(names=names, iou_thr=0.5, conf_thr=0.5)
    
    gt_anns = [
        {'category_id': 1, 'bbox': [10, 10, 20, 20]},
        {'category_id': 2, 'bbox': [50, 50, 20, 20]}
    ]
    pred_anns = [
        # Error: class2 predicted for class1 GT (high IoU)
        {'category_id': 2, 'bbox': [12, 12, 18, 18], 'score': 0.9},
        # TP for class2
        {'category_id': 2, 'bbox': [52, 52, 18, 18], 'score': 0.9},
        # Traditional FP (class3 with no GT)
        {'category_id': 3, 'bbox': [100, 100, 30, 30], 'score': 0.8}
    ]
    metrics.process_image(gt_anns, pred_anns)
    
    # Expected matrix: {1:0, 2:1, 3:2}, background_idx=3
    expected_matrix = np.zeros((4, 4), dtype=int)
    expected_matrix[0, 1] = 1  # GT class0 predicted as class1
    expected_matrix[1, 1] = 1  # GT class1 predicted correctly
    expected_matrix[3, 2] = 1  # FP class2 (unmatched prediction)
    
    assert np.array_equal(metrics.multiclass_matrix, expected_matrix), (
        f"Incorrect confusion matrix.\nExpected:\n{expected_matrix}\nGot:\n{metrics.matrix}"
    )
    
    # Validate per-class statistics
    metrics_result = metrics.compute_metrics()
    # Class1: FN (GT not detected)
    assert metrics_result[1]['tp'] == 0
    assert metrics_result[1]['fp'] == 0
    assert metrics_result[1]['fn'] == 1
    assert metrics_result[1]['support'] == 1
    
    # Class2: 1 TP + 1 FP (from classification error)
    assert metrics_result[2]['tp'] == 1
    assert metrics_result[2]['fp'] == 1
    assert metrics_result[2]['fn'] == 0
    assert metrics_result[2]['support'] == 1
    
    # Class3: Traditional FP
    assert metrics_result[3]['tp'] == 0
    assert metrics_result[3]['fp'] == 1
    assert metrics_result[3]['fn'] == 0
    assert metrics_result[3]['support'] == 0
    
    # Global metrics
    global_metrics = metrics_result['global']
    assert global_metrics['precision'] == 1/3
    assert global_metrics['recall'] == 0.5
    expected_f1 = 2 * (1/3) * 0.5 / ((1/3) + 0.5)
    assert abs(global_metrics['f1'] - expected_f1) < 1e-6
    assert global_metrics['support'] == 2

def test_iscrowd_handling():
    """
    Tests correct handling of 'iscrowd' annotations according to COCO evaluation rules.
    
    Verifies:
    - Crowd annotations (iscrowd=1) are excluded from matching and do not contribute to false negatives.
    - Predictions matching crowd regions are not penalized as false positives.
    - Normal annotations (iscrowd=0) are evaluated normally.
    
    Scenario:
      Image contains:
        - 1 normal ground truth (class1, bbox1)
        - 1 crowd ground truth (class2, bbox2)
        - Predictions:
            1. Correct match for normal GT (class1, bbox1) - should be TP
            2. Detection matching crowd region (class2, bbox2) - should be ignored
            3. Detection with no match (class3) - should be FP
    """
    # Ground truths
    gt_anns = [
        {'category_id': 1, 'bbox': [10, 10, 20, 20], 'iscrowd': 0},  # Normal object
        {'category_id': 2, 'bbox': [50, 50, 20, 20], 'iscrowd': 1}   # Crowd region
    ]
    
    # Predictions
    pred_anns = [
        # TP: Correct match for normal GT (IoU > threshold)
        {'category_id': 1, 'bbox': [12, 12, 18, 18], 'score': 0.9},
        
        # Should be ignored: Matches crowd region (same class)
        {'category_id': 2, 'bbox': [52, 52, 18, 18], 'score': 0.8},
        
        # FP: No matching GT
        {'category_id': 3, 'bbox': [100, 100, 30, 30], 'score': 0.7}
    ]
    
    metrics = DetectionMetrics(
        names={1: 'class1', 2: 'class2', 3: 'class3'},
        iou_thr=0.5,
        conf_thr=0.5
    )
    metrics.process_image(gt_anns, pred_anns)
    results = metrics.compute_metrics()
    
    # Validate metrics
    assert results[1]['tp'] == 1, "Class1 should have 1 TP"
    assert results[1]['fp'] == 0, "Class1 should have 0 FP"
    assert results[1]['fn'] == 0, "Class1 should have 0 FN"
    assert results[1]['support'] == 1, "Class1 support should be 1"
    
    assert results[2]['tp'] == 0, "Class2 should have 0 TP (crowd ignored)"
    assert results[2]['fp'] == 0, "Class2 should have 0 FP (crowd match ignored)"
    assert results[2]['fn'] == 0, "Class2 should have 0 FN (crowd not counted)"
    assert results[2]['support'] == 0, "Class2 support should be 0 (crowd excluded)"
    
    assert results[3]['tp'] == 0, "Class3 should have 0 TP"
    assert results[3]['fp'] == 1, "Class3 should have 1 FP"
    assert results[3]['fn'] == 0, "Class3 should have 0 FN"
    assert results[3]['support'] == 0, "Class3 support should be 0"
    
    # Verify confusion matrix
    # Expected: {1:0, 2:1, 3:2}, background_idx=3
    expected_matrix = np.array([
        [1, 0, 0, 0],  # class1: 1 TP
        [0, 0, 0, 0],  # class2: no entries
        [0, 0, 0, 0],  # class3: no FNs (since no GTs)
        [0, 0, 1, 0]   # background row: FP for class3 (column 2)
    ])
    assert np.array_equal(metrics.matrix, expected_matrix), \
        "Confusion matrix mismatch"
    
@pytest.mark.coco_path
def test_precision_recall_curve_basic():
    """Test basic precision-recall curve calculation with single class.
    
    Verifies:
    - Correct precision and recall values at different confidence thresholds
    - Proper AP calculation
    - Handling of TP, FP, FN cases
    """
    # Ground truth for 2 images
    all_gts = [
        [{'category_id': 1, 'bbox': [10, 10, 20, 20]}],  # Image 1
        [{'category_id': 1, 'bbox': [50, 50, 30, 30]}]   # Image 2
    ]
    
    # Predictions with varying confidence
    all_preds = [
        [  # Image 1 predictions
            {'category_id': 1, 'bbox': [12, 12, 18, 18], 'score': 0.9},  # TP (IoU=0.81)
            {'category_id': 1, 'bbox': [15, 15, 15, 15], 'score': 0.8}   # TP (IoU=0.75)
        ],
        [  # Image 2 predictions
            {'category_id': 1, 'bbox': [55, 55, 25, 25], 'score': 0.7},  # TP (IoU=0.69)
            {'category_id': 1, 'bbox': [100, 100, 30, 30], 'score': 0.6}  # FP
        ]
    ]
    
    # Compute PR curve
    result = compute_precision_recall_curve(all_gts, all_preds, iou_threshold=0.5)
    exp_ap, exp_per_class = _compute_oracle_from_batches(all_gts, all_preds, iou_threshold=0.5)
    
    # Validate results
    assert len(result['precision']) >= 1
    assert len(result['recall']) >= 1
    assert len(result['thresholds']) >= 1

    # AP and per-class AP must match COCO API oracle.
    assert float(result['ap']) == pytest.approx(exp_ap, rel=1e-6)
    assert float(result['per_class'][1]) == pytest.approx(exp_per_class[1], rel=1e-6)

@pytest.mark.coco_path
def test_precision_recall_curve_multiclass():
    """Test precision-recall curve with multiple classes.
    
    Verifies:
    - Per-class AP calculation
    - Correct handling of class-specific metrics
    - Global AP aggregation
    """
    # Ground truth
    all_gts = [
        [
            {'category_id': 1, 'bbox': [10, 10, 20, 20]},
            {'category_id': 2, 'bbox': [50, 50, 30, 30]}
        ]
    ]
    
    # Predictions
    all_preds = [
        [
            {'category_id': 1, 'bbox': [12, 12, 18, 18], 'score': 0.9},  # TP class1
            {'category_id': 2, 'bbox': [55, 55, 25, 25], 'score': 0.8},   # TP class2
            {'category_id': 1, 'bbox': [100, 100, 30, 30], 'score': 0.7}  # FP class1
        ]
    ]
    
    # Compute PR curve
    result = compute_precision_recall_curve(all_gts, all_preds, iou_threshold=0.5)
    exp_ap, exp_per_class = _compute_oracle_from_batches(all_gts, all_preds, iou_threshold=0.5)

    # Validate per-class/global AP numerically against oracle.
    assert float(result['per_class'][1]) == pytest.approx(exp_per_class[1], rel=1e-6)
    assert float(result['per_class'][2]) == pytest.approx(exp_per_class[2], rel=1e-6)
    assert float(result['ap']) == pytest.approx(exp_ap, rel=1e-6)

@pytest.mark.coco_path
def test_precision_recall_curve_crowd_annotations():
    """Test handling of crowd annotations.
    
    Verifies:
    - Crowd annotations are excluded from matching
    - Predictions matching crowd regions don't count as FP
    - Recall calculation ignores crowd GTs
    """
    all_gts = [
        [
            {'category_id': 1, 'bbox': [10, 10, 20, 20], 'iscrowd': 0},  # Normal
            {'category_id': 1, 'bbox': [50, 50, 30, 30], 'iscrowd': 1}   # Crowd
        ]
    ]
    
    all_preds = [
        [
            {'category_id': 1, 'bbox': [12, 12, 18, 18], 'score': 0.9},  # TP
            {'category_id': 1, 'bbox': [52, 52, 28, 28], 'score': 0.8}   # Ignored (crowd match)
        ]
    ]
    
    result = compute_precision_recall_curve(all_gts, all_preds)
    exp_ap, exp_per_class = _compute_oracle_from_batches(all_gts, all_preds, iou_threshold=0.5)
    
    # With COCOeval, prediction on crowd region is ignored (not FP), so precision remains ~1.0
    assert result['precision'][-1] == pytest.approx(1.0, abs=0.05)
    assert result['recall'][-1] == pytest.approx(1.0, abs=0.01)
    assert float(result['ap']) == pytest.approx(exp_ap, rel=1e-6)
    assert float(result['per_class'][1]) == pytest.approx(exp_per_class[1], rel=1e-6)

@pytest.mark.coco_path
def test_precision_recall_curve_empty_inputs():
    """Test edge cases with empty inputs.
    
    Verifies:
    - Graceful handling of empty ground truth
    - Graceful handling of empty predictions
    - Zero AP when no valid detections
    """
    # Case 1: No ground truth
    result1 = compute_precision_recall_curve([], [[]])
    assert result1['ap'] == 0.0
    assert len(result1['precision']) == 0
    
    # Case 2: No predictions
    all_gts = [[{'category_id': 1, 'bbox': [10, 10, 20, 20]}]]
    all_preds = [[]]
    result2 = compute_precision_recall_curve(all_gts, all_preds)
    assert result2['ap'] == 0.0
    
    # Case 3: Valid data but no matches
    all_preds = [[{'category_id': 1, 'bbox': [100, 100, 30, 30], 'score': 0.9}]]
    result3 = compute_precision_recall_curve(all_gts, all_preds)
    assert result3['ap'] == 0.0
    if result3['precision'].size > 0:
        assert result3['precision'][-1] == 0.0

def test_pr_curves_early_return():
    """
    Tests graceful handling of incomplete COCO evaluation data.

    Simulates scenarios where COCO evaluation fails to generate required data
    (e.g., empty COCO objects) by verifying:
    1. The function returns without raising exceptions
    2. No PR curves are generated (pr_curves remains empty)
    """

    metrics = DetectionMetrics(
        names={1: 'class1'},
        gt_coco=COCO(), 
        predictions_coco=COCO(),  
        store_pr_curves=True
    )
    
    dummy_metrics = {
        'global': {'mAP': 0.5},
        1: {'ap': 0.6}
    }
    
    metrics._compute_pr_curves(dummy_metrics)    
    assert not metrics.pr_curves

@pytest.mark.coco_path
def test_precision_recall_curve_class_without_ground_truth():
    """Test per-class AP calculation for    classes with no ground truth instances.
    
    Verifies:
    - Classes with 0 ground truths receive AP=0.0
    - Does not affect AP calculation for other classes
    - Proper handling of 'per_class' dictionary entries
    """
    # Ground truth - class 1 only
    all_gts = [
        [{'category_id': 1, 'bbox': [10, 10, 20, 20]}],  # Image 1
        [{'category_id': 1, 'bbox': [50, 50, 30, 30]}]   # Image 2
    ]
    
    # Predictions include class 2 (no ground truth) and class 1
    all_preds = [
        [  # Image 1 predictions
            {'category_id': 1, 'bbox': [12, 12, 18, 18], 'score': 0.9},  # TP class1
            {'category_id': 2, 'bbox': [100, 100, 30, 30], 'score': 0.8}  # FP class2
        ],
        [  # Image 2 predictions
            {'category_id': 1, 'bbox': [52, 52, 28, 28], 'score': 0.7},  # TP class1
            {'category_id': 2, 'bbox': [150, 150, 30, 30], 'score': 0.6}  # FP class2
        ]
    ]
    
    # Compute PR curve
    result = compute_precision_recall_curve(all_gts, all_preds, iou_threshold=0.5)
    
    # Validate per-class results
    assert 1 in result['per_class']
    assert 2 in result['per_class']
    
    # Class 1 should have normal AP
    assert result['per_class'][1] > 0
    
    # Class 2 (no ground truth) must have AP=0.0
    assert result['per_class'][2] == 0.0
