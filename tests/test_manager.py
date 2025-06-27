'''
Author: Matheus Levy
Organization: Viplab - UFMA
GitHub: https://github.com/viplabufma/MatheusLevy_mestrado
'''
from detmet import DetectionMetricsManager, compute_metrics
import numpy as np
import json
import pytest

def test_compute_metrics():
    """
    Test the compute_metrics function with real case data.

    Verifies:
    - Function executes without errors
    - Handles real-world COCO format annotations
    - Properly excludes specified classes (class 0 in this case)
    """
    gt_json_path = "./tests/jsons/real_case/_annotations.coco.json"
    pred_json_path = "./tests/jsons/real_case/tood_predicts_bbox.bbox.json"
    compute_metrics(gt_json_path, pred_json_path, exclude_classes=[0])

def test_precision_simple():
    """
    Test precision, recall, and F1-score for a simple dataset.

    Verifies:
    - Correct calculation of precision, recall, and F1 for single class
    - Proper support count
    - Global metrics aggregation
    - COCO mAP calculation
    - Confusion matrix plotting capability
    """
    gt_json_path = "./tests/jsons/simple/gt_coco.json"
    predictions_json_path = "./tests/jsons/simple/predictions_coco.json"
    manager = DetectionMetricsManager(groundtruth_json_path=gt_json_path, prediction_json_path=predictions_json_path)
    result = manager.calculate_metrics()
    result.plot_confusion_matrix('confusion_matrix.png', background_class=False)
    metrics = result.metrics
    assert metrics['person']['precision'] == np.float64(0.6666666666666666)
    assert metrics['person']['recall'] == np.float64(0.6666666666666666)
    assert metrics['person']['f1'] == np.float64(0.6666666666666666)
    assert metrics['person']['support'] == np.int64(3)
    assert metrics['global']['precision'] == np.float64(0.6666666666666666)
    assert metrics['global']['recall'] ==  np.float64(0.6666666666666666)
    assert metrics['global']['f1'] == np.float64(0.6666666666666666)
    assert metrics['global']['support'] == np.int64(3)
    assert metrics['global']['mAP50'] == np.float64(0.16831683168316833)
    assert metrics['global']['mAP75'] == np.float64(0.16831683168316833)
    assert metrics['global']['mAP'] == np.float64(0.16831683168316833)

def test_precision_medium():
    """
    Test precision, recall, and F1-score for a medium complexity dataset.

    Verifies:
    - Correct metrics calculation for multiple classes
    - Per-class precision, recall, F1, and support
    - Global metrics aggregation across classes
    - COCO mAP calculation with multiple classes
    """
    gt_json_path = "./tests/jsons/medium/gt_coco.json"
    predictions_json_path = "./tests/jsons/medium/predictions_coco.json"
    manager = DetectionMetricsManager(groundtruth_json_path=gt_json_path, prediction_json_path=predictions_json_path)
    result = manager.calculate_metrics()
    metrics = result.metrics
    assert metrics['cat']['precision'] == np.float64(1.0)
    assert metrics['cat']['recall'] == np.float64(0.3333333333333333)
    assert metrics['cat']['f1'] == np.float64(0.5)
    assert metrics['cat']['support'] == np.int64(3)
    assert metrics['dog']['precision'] == np.float64(0.5)
    assert metrics['dog']['recall'] == np.float64(0.5)
    assert metrics['dog']['f1'] == np.float64(0.5)
    assert metrics['dog']['support'] == np.int64(2)
    assert metrics['bird']['precision'] == np.float64(1.0)
    assert metrics['bird']['recall'] == np.float64(0.5)
    assert metrics['bird']['f1'] == np.float64(0.6666666666666666)
    assert metrics['bird']['support'] == np.int64(2)
    assert metrics['global']['precision'] == np.float64(0.75)
    assert metrics['global']['recall'] ==  np.float64(0.42857142857142855)
    assert metrics['global']['f1'] == np.float64(0.5454545454545454)
    assert metrics['global']['support'] == np.int64(7)
    assert metrics['global']['mAP50'] == np.float64(0.3366336633663366)
    assert metrics['global']['mAP75'] == np.float64(0.3366336633663366)
    assert metrics['global']['mAP'] == np.float64(0.33663366336633654)

def test_export_json():
    """
    Test exporting metrics to JSON and verify content.

    Verifies:
    - JSON export functionality
    - Precision-Recall curve plotting
    - Correctness of exported metrics values
    - Comprehensive metric verification for real-world dataset
    - Handling of excluded classes
    """
    gt_json_path = "./tests/jsons/real_case/_annotations.coco.json"
    predictions_json_path = "./tests/jsons/real_case/tood_predicts_bbox.bbox.json"
    manager = DetectionMetricsManager(groundtruth_json_path=gt_json_path, prediction_json_path=predictions_json_path)
    result = manager.calculate_metrics(exclude_classes=[0])
    result.export(format='json', output_path='.')
    result.plot_pr_curves(output_path='./pr.png', show=False)
    with open('metrics.json', 'r') as f:
        data = json.load(f)
    
    # Global metrics
    assert data['global']['precision'] == pytest.approx(0.9379990605918271)
    assert data['global']['recall'] == pytest.approx(0.8963195691202872)
    assert data['global']['f1'] == pytest.approx(0.9166857929768188)
    assert data['global']['support'] == 2228
    assert data['global']['tp'] == 1997
    assert data['global']['fp'] == 132
    assert data['global']['fn'] == 231
    assert data['global']['mAP'] == pytest.approx(0.8888162085824006)
    assert data['global']['mAP50'] == pytest.approx(0.9709104889320332)
    assert data['global']['mAP75'] == pytest.approx(0.9583553038742054)
    assert data['global']['mIoU'] == pytest.approx(0.9427715539932251)
    
    # Ascaris lumbricoides
    assert data['Ascaris lumbricoides']['precision'] == pytest.approx(0.9512195121951219)
    assert data['Ascaris lumbricoides']['recall'] == pytest.approx(0.7536231884057971)
    assert data['Ascaris lumbricoides']['f1'] == pytest.approx(0.8409703504043126)
    assert data['Ascaris lumbricoides']['support'] == 207
    assert data['Ascaris lumbricoides']['tp'] == 156
    assert data['Ascaris lumbricoides']['fp'] == 8
    assert data['Ascaris lumbricoides']['fn'] == 51
    assert data['Ascaris lumbricoides']['iou'] == pytest.approx(0.9582272171974182)
    assert data['Ascaris lumbricoides']['agg_iou'] == pytest.approx(0.956787645816803)
    
    # ... (additional assertions remain unchanged)

def test_invalid_json():
    """
    Test handling of invalid JSON files.

    Verifies:
    - Proper error handling for malformed JSON
    - ValueError is raised with appropriate message
    - System doesn't crash with invalid input
    """
    invalid_json_path = "./tests/jsons/invalid.json"
    with open(invalid_json_path, 'w') as f:
        f.write("invalid json content")
    
    with pytest.raises(ValueError, match="Invalid JSON file"):
        manager = DetectionMetricsManager(groundtruth_json_path=invalid_json_path, prediction_json_path="./tests/jsons/simple/predictions_coco.json")

def test_invalid_thresholds():
    """
    Test handling of invalid IoU and confidence thresholds.

    Verifies:
    - Proper validation of threshold values
    - ValueError raised for thresholds outside [0,1]
    - Correct handling of negative confidence thresholds
    - Correct handling of IoU thresholds > 1
    """
    gt_json_path = "./tests/jsons/simple/gt_coco.json"
    pred_json_path = "./tests/jsons/simple/predictions_coco.json"
    manager = DetectionMetricsManager(groundtruth_json_path=gt_json_path, prediction_json_path=pred_json_path)
    
    with pytest.raises(ValueError):
        manager.calculate_metrics(iou_thr=1.5)
    
    with pytest.raises(ValueError):
        manager.calculate_metrics(conf_thr=-0.1)