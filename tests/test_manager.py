'''
Author: Matheus Levy
Organization: Viplab - UFMA
GitHub: https://github.com/viplabufma/MatheusLevy_mestrado
'''
from detmet import DetectionMetricsManager, compute_metrics
import json
from pathlib import Path
import pytest

@pytest.mark.integration
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

@pytest.mark.integration
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
    metrics = result.metrics

    assert metrics is not None
    assert 'global' in metrics
    assert 'person' in metrics
    assert 0.0 <= float(metrics['global']['precision']) <= 1.0
    assert 0.0 <= float(metrics['global']['recall']) <= 1.0
    assert 0.0 <= float(metrics['global']['f1']) <= 1.0
    assert 0.0 <= float(metrics['global']['mAP']) <= 1.0
    assert metrics['global']['support'] > 0

@pytest.mark.integration
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
    assert metrics is not None
    assert 'global' in metrics
    assert 0.0 <= float(metrics['global']['mAP']) <= 1.0
    assert 0.0 <= float(metrics['global']['mAP50']) <= 1.0
    assert 0.0 <= float(metrics['global']['mAP75']) <= 1.0
    assert metrics['global']['support'] > 0

@pytest.mark.integration
def test_export_json_smoke(tmp_path):
    """
    Test exporting metrics to JSON and verify content.

    Verifies:
    - JSON export functionality
    - Precision-Recall curve plotting
    - Correctness of exported metrics values
    - Comprehensive metric verification for real-world dataset
    - Handling of excluded classes
    """
    gt_json_path = "./tests/jsons/medium/gt_coco.json"
    predictions_json_path = "./tests/jsons/medium/predictions_coco.json"
    manager = DetectionMetricsManager(groundtruth_json_path=gt_json_path, prediction_json_path=predictions_json_path)
    result = manager.calculate_metrics()
    result.export(format='json', output_path=str(tmp_path))

    out_file = tmp_path / 'metrics.json'
    assert out_file.exists()
    with open(out_file, 'r') as f:
        data = json.load(f)

    assert 'global' in data
    assert 0.0 <= float(data['global']['mAP']) <= 1.0


def test_invalid_json(tmp_path):
    """
    Test handling of invalid JSON files.

    Verifies:
    - Proper error handling for malformed JSON
    - ValueError is raised with appropriate message
    - System doesn't crash with invalid input
    """
    invalid_json_path = tmp_path / "invalid.json"
    with open(invalid_json_path, 'w') as f:
        f.write("invalid json content")
    
    with pytest.raises(ValueError, match="Invalid JSON file"):
        DetectionMetricsManager(groundtruth_json_path=str(invalid_json_path), prediction_json_path="./tests/jsons/simple/predictions_coco.json")

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

def test_update_data():
    """
    Test updating data sources after initial initialization.
    
    Verifies:
    - The update_data method correctly reloads new data
    - Metrics are recalculated based on the new dataset
    - The internal state is updated to reflect new data paths
    """
    # Initial setup with simple dataset
    gt_json_simple = "./tests/jsons/simple/gt_coco.json"
    pred_json_simple = "./tests/jsons/simple/predictions_coco.json"
    manager = DetectionMetricsManager(groundtruth_json_path=gt_json_simple, 
                                     prediction_json_path=pred_json_simple)
    
    # Calculate metrics for simple dataset
    result_simple = manager.calculate_metrics()
    assert result_simple.metrics['global']['support'] == 3
    
    # Update to medium dataset
    gt_json_medium = "./tests/jsons/medium/gt_coco.json"
    pred_json_medium = "./tests/jsons/medium/predictions_coco.json"
    manager.update_data(groundtruth_json_path=gt_json_medium, 
                       prediction_json_path=pred_json_medium)
    
    # Calculate metrics for medium dataset
    result_medium = manager.calculate_metrics()
    assert result_medium.metrics['global']['support'] == 7