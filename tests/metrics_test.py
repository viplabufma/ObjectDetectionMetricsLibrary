import numpy as np
from metrics import DetectionMetrics

def test_multi_image_processing():
    """Tests processing of multiple images with accumulated statistics"""
    # Image 1
    gt_anns1 = [{'category_id': 1, 'bbox': [10, 10, 20, 20]}]
    pred_anns1 = [{'category_id': 1, 'bbox': [12, 12, 18, 18], 'score': 0.9}]
    
    # Image 2
    gt_anns2 = [{'category_id': 2, 'bbox': [50, 50, 20, 20]}]
    pred_anns2 = [
        {'category_id': 2, 'bbox': [52, 52, 18, 18], 'score': 0.9},
        {'category_id': 3, 'bbox': [100, 100, 30, 30], 'score': 0.8}
    ]
    
    # Process images
    metrics = DetectionMetrics(names={1: 'class1', 2: 'class2', 3: 'class3'})
    metrics.process_image(gt_anns1, pred_anns1)
    metrics.process_image(gt_anns2, pred_anns2)
    
    # Verify accumulated confusion matrix
    # With global mapping: {1: 0, 2: 1, 3: 2}, background_idx = 3
    expected_matrix = np.zeros((4, 4), dtype=int)
    expected_matrix[0, 0] = 1  # TP class 1 (index 0)
    expected_matrix[1, 1] = 1  # TP class 2 (index 1)
    expected_matrix[3, 2] = 1  # FP class 3 (index 2)
    
    assert np.array_equal(metrics.matrix, expected_matrix)
    
    # Verify statistics
    metrics_result = metrics.compute_metrics()
    assert metrics_result[1]['precision'] == 1.0
    assert metrics_result[1]['recall'] == 1.0
    assert metrics_result[2]['precision'] == 1.0
    assert metrics_result[2]['recall'] == 1.0
    assert metrics_result[3]['precision'] == 0.0  # 0 TP, 1 FP
    assert metrics_result[3]['recall'] == 0.0     # 0 TP, 0 FN (no GT for class 3)

def test_single_image_basic():
    """Tests basic functionality with a single image"""
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
    
    # Both classes should have precision and recall = 1.0
    assert result[1]['precision'] == 1.0
    assert result[1]['recall'] == 1.0
    assert result[2]['precision'] == 1.0
    assert result[2]['recall'] == 1.0

def test_no_predictions():
    """Tests case when there are no predictions"""
    gt_anns = [
        {'category_id': 1, 'bbox': [10, 10, 20, 20]},
        {'category_id': 2, 'bbox': [50, 50, 30, 30]}
    ]
    pred_anns = []
    
    metrics = DetectionMetrics(names={1: 'person', 2: 'car'})
    metrics.process_image(gt_anns, pred_anns)
    
    result = metrics.compute_metrics()
    
    # No predictions: precision = 0, recall = 0
    assert result[1]['precision'] == 0.0
    assert result[1]['recall'] == 0.0
    assert result[2]['precision'] == 0.0
    assert result[2]['recall'] == 0.0
    assert result[1]['support'] == 1
    assert result[2]['support'] == 1

def test_no_ground_truth():
    """Tests case when there is no ground truth"""
    gt_anns = []
    pred_anns = [
        {'category_id': 1, 'bbox': [12, 12, 18, 18], 'score': 0.9},
        {'category_id': 2, 'bbox': [52, 52, 28, 28], 'score': 0.8}
    ]
    
    metrics = DetectionMetrics(names={1: 'person', 2: 'car'})
    metrics.process_image(gt_anns, pred_anns)
    
    result = metrics.compute_metrics()
    
    # No GT: precision = 0 (all are FP), recall = 0 (no GT)
    assert result[1]['precision'] == 0.0
    assert result[1]['recall'] == 0.0
    assert result[2]['precision'] == 0.0
    assert result[2]['recall'] == 0.0
    assert result[1]['support'] == 0
    assert result[2]['support'] == 0

def test_confidence_threshold():
    """Tests confidence threshold filtering"""
    gt_anns = [{'category_id': 1, 'bbox': [10, 10, 20, 20]}]
    pred_anns = [
        {'category_id': 1, 'bbox': [12, 12, 18, 18], 'score': 0.9},  # Above threshold
        {'category_id': 1, 'bbox': [15, 15, 15, 15], 'score': 0.3}   # Below threshold
    ]
    
    metrics = DetectionMetrics(names={1: 'person'}, conf_thr=0.5)
    metrics.process_image(gt_anns, pred_anns)
    
    result = metrics.compute_metrics()
    
    # Only one prediction should be considered
    assert result[1]['precision'] == 1.0
    assert result[1]['recall'] == 1.0

def test_iou_threshold():
    """Tests IoU threshold for matching detections"""
    gt_anns = [{'category_id': 1, 'bbox': [10, 10, 20, 20]}]
    pred_anns = [
        {'category_id': 1, 'bbox': [30, 30, 20, 20], 'score': 0.9}  # Very low IoU
    ]
    
    metrics = DetectionMetrics(names={1: 'person'}, iou_thr=0.5)
    metrics.process_image(gt_anns, pred_anns)
    
    result = metrics.compute_metrics()
    
    # Low IoU should result in FP and FN
    assert result[1]['precision'] == 0.0  # 0 TP, 1 FP
    assert result[1]['recall'] == 0.0     # 0 TP, 1 FN

def test_class_mismatch():
    """Tests when prediction has different class than ground truth"""
    gt_anns = [{'category_id': 1, 'bbox': [10, 10, 20, 20]}]
    pred_anns = [{'category_id': 2, 'bbox': [12, 12, 18, 18], 'score': 0.9}]
    
    metrics = DetectionMetrics(names={1: 'person', 2: 'car'})
    metrics.process_image(gt_anns, pred_anns)
    
    result = metrics.compute_metrics()
    
    # Different classes: FP for class 2, FN for class 1
    assert result[1]['precision'] == 0.0  # No TP for class 1
    assert result[1]['recall'] == 0.0     # FN for class 1
    assert result[2]['precision'] == 0.0  # FP for class 2
    assert result[2]['recall'] == 0.0     # No GT for class 2

def test_multiple_predictions_same_gt():
    """Tests multiple predictions for the same ground truth"""
    gt_anns = [{'category_id': 1, 'bbox': [10, 10, 20, 20]}]
    pred_anns = [
        {'category_id': 1, 'bbox': [12, 12, 18, 18], 'score': 0.9},
        {'category_id': 1, 'bbox': [11, 11, 19, 19], 'score': 0.8}
    ]
    
    metrics = DetectionMetrics(names={1: 'person'})
    metrics.process_image(gt_anns, pred_anns)
    
    result = metrics.compute_metrics()
    
    # Only one prediction should be matched, the other is FP
    assert result[1]['precision'] == 0.5  # 1 TP, 1 FP
    assert result[1]['recall'] == 1.0     # 1 TP, 0 FN

def test_bbox_iou_calculation():
    """Tests specific IoU calculation between bounding boxes"""
    metrics = DetectionMetrics(names={1: 'test'})
    
    # Identical boxes
    iou1 = metrics.bbox_iou([10, 10, 20, 20], [10, 10, 20, 20])
    assert iou1 == 1.0
    
    # Non-overlapping boxes
    iou2 = metrics.bbox_iou([10, 10, 20, 20], [50, 50, 20, 20])
    assert iou2 == 0.0
    
    # Partially overlapping boxes
    iou3 = metrics.bbox_iou([10, 10, 20, 20], [20, 20, 20, 20])
    assert iou3 > 0.0 and iou3 < 1.0

def test_results_dict_property():
    """Tests the results_dict property for comprehensive metrics"""
    gt_anns = [{'category_id': 1, 'bbox': [10, 10, 20, 20]}]
    pred_anns = [{'category_id': 1, 'bbox': [12, 12, 18, 18], 'score': 0.9}]
    
    metrics = DetectionMetrics(names={1: 'person', 2: 'car'})
    metrics.process_image(gt_anns, pred_anns)
    
    results = metrics.results_dict
    
    # Verify expected keys are present
    expected_keys = [
        'precision/person', 'recall/person', 'f1/person', 'support/person',
        'precision/car', 'recall/car', 'f1/car', 'support/car'
    ]
    
    for key in expected_keys:
        assert key in results

def test_reset_functionality():
    """Tests the reset functionality"""
    gt_anns = [{'category_id': 1, 'bbox': [10, 10, 20, 20]}]
    pred_anns = [{'category_id': 1, 'bbox': [12, 12, 18, 18], 'score': 0.9}]
    
    metrics = DetectionMetrics(names={1: 'person'})
    metrics.process_image(gt_anns, pred_anns)
    
    # Verify there is data
    assert metrics.matrix is not None
    assert len(metrics.stats) > 0
    
    # Reset
    metrics.reset()
    
    # Verify reset state
    assert metrics.matrix is None
    assert len(metrics.stats) == 0
    assert len(metrics.class_map) == 0

def test_edge_case_empty_inputs():
    """Tests edge case with empty inputs"""
    metrics = DetectionMetrics(names={1: 'person'})
    metrics.process_image([], [])
    
    result = metrics.compute_metrics()
    
    # With empty inputs, metrics should be 0
    assert result[1]['precision'] == 0.0
    assert result[1]['recall'] == 0.0
    assert result[1]['support'] == 0

def test_three_images_metrics():
    """Tests processing of 3 images with full metrics calculation including global metrics"""
    # Image 1
    gt_anns1 = [
        {'category_id': 1, 'bbox': [10, 10, 20, 20]},
        {'category_id': 2, 'bbox': [50, 50, 20, 20]}
    ]
    pred_anns1 = [
        {'category_id': 1, 'bbox': [12, 12, 18, 18], 'score': 0.9},  # TP class1
        {'category_id': 2, 'bbox': [52, 52, 18, 18], 'score': 0.9},  # TP class2
        {'category_id': 3, 'bbox': [100, 100, 30, 30], 'score': 0.8} # FP class3
    ]
    
    # Image 2
    gt_anns2 = [
        {'category_id': 1, 'bbox': [10, 10, 20, 20]},
        {'category_id': 3, 'bbox': [50, 50, 20, 20]}
    ]
    pred_anns2 = [
        {'category_id': 1, 'bbox': [12, 12, 18, 18], 'score': 0.9},  # TP class1
        {'category_id': 2, 'bbox': [100, 100, 30, 30], 'score': 0.8}, # FP class2
        {'category_id': 3, 'bbox': [52, 52, 18, 18], 'score': 0.9}   # TP class3
    ]
    
    # Image 3
    gt_anns3 = [
        {'category_id': 2, 'bbox': [10, 10, 20, 20]},
        {'category_id': 3, 'bbox': [50, 50, 20, 20]}
    ]
    pred_anns3 = [
        {'category_id': 1, 'bbox': [100, 100, 30, 30], 'score': 0.8}, # FP class1
        {'category_id': 2, 'bbox': [12, 12, 18, 18], 'score': 0.9},   # TP class2
        {'category_id': 3, 'bbox': [52, 52, 18, 18], 'score': 0.6},   # TP class3
        {'category_id': 3, 'bbox': [55, 55, 15, 15], 'score': 0.3}    # Ignored (conf < thr)
    ]
    
    # Process the 3 images
    metrics = DetectionMetrics(names={1: 'class1', 2: 'class2', 3: 'class3'})
    metrics.process_image(gt_anns1, pred_anns1)
    metrics.process_image(gt_anns2, pred_anns2)
    metrics.process_image(gt_anns3, pred_anns3)
    
    # 1. Verify final confusion matrix
    # Mapping: {1:0, 2:1, 3:2}, background_idx=3
    expected_matrix = np.zeros((4, 4), dtype=int)
    # TPs on diagonal
    expected_matrix[0, 0] = 2  # class1 TP (2 images)
    expected_matrix[1, 1] = 2  # class2 TP (img1 and img3)
    expected_matrix[2, 2] = 2  # class3 TP (img2 and img3)
    # FPs (background -> prediction)
    expected_matrix[3, 0] = 1  # FP class1 (img3)
    expected_matrix[3, 1] = 1  # FP class2 (img2)
    expected_matrix[3, 2] = 1  # FP class3 (img1)
    
    assert np.array_equal(metrics.matrix, expected_matrix), \
        f"Incorrect confusion matrix.\nExpected:\n{expected_matrix}\nGot:\n{metrics.matrix}"
    
    # 2. Verify per-class metrics
    metrics_result = metrics.compute_metrics()
    
    # Class 1: 2 TP, 1 FP, 0 FN
    assert metrics_result[1]['precision'] == 2/3, f"Class1 precision: expected {2/3}, got {metrics_result[1]['precision']}"
    assert metrics_result[1]['recall'] == 1.0, f"Class1 recall: expected 1.0, got {metrics_result[1]['recall']}"
    assert metrics_result[1]['f1'] == (2 * (2/3) * 1.0) / ((2/3) + 1.0), "Incorrect F1 class1"
    assert metrics_result[1]['support'] == 2, "Incorrect support class1"
    
    # Class 2: 2 TP, 1 FP, 0 FN
    assert metrics_result[2]['precision'] == 2/3, f"Class2 precision: expected {2/3}, got {metrics_result[2]['precision']}"
    assert metrics_result[2]['recall'] == 1.0, f"Class2 recall: expected 1.0, got {metrics_result[2]['recall']}"
    assert metrics_result[2]['f1'] == (2 * (2/3) * 1.0) / ((2/3) + 1.0), "Incorrect F1 class2"
    assert metrics_result[2]['support'] == 2, "Incorrect support class2"
    
    # Class 3: 2 TP, 1 FP, 0 FN
    assert metrics_result[3]['precision'] == 2/3, f"Class3 precision: expected {2/3}, got {metrics_result[3]['precision']}"
    assert metrics_result[3]['recall'] == 1.0, f"Class3 recall: expected 1.0, got {metrics_result[3]['recall']}"
    assert metrics_result[3]['f1'] == (2 * (2/3) * 1.0) / ((2/3) + 1.0), "Incorrect F1 class3"
    assert metrics_result[3]['support'] == 2, "Incorrect support class3"
    
    # 3. Verify global metrics
    # Total TP = 2 (class1) + 2 (class2) + 2 (class3) = 6
    # Total FP = 1 (class1) + 1 (class2) + 1 (class3) = 3
    # Total FN = 0 (all GT detected)
    global_metrics = metrics_result['global']
    assert global_metrics['precision'] == 6/(6+3), f"Global precision: expected {6/9}, got {global_metrics['precision']}"
    assert global_metrics['recall'] == 1.0, f"Global recall: expected 1.0, got {global_metrics['recall']}"
    expected_f1 = 2 * (6/9 * 1.0) / (6/9 + 1.0) if (6/9 + 1.0) > 0 else 0.0
    assert abs(global_metrics['f1'] - expected_f1) < 1e-6, f"Global F1: expected {expected_f1}, got {global_metrics['f1']}"
    assert global_metrics['support'] == 6, f"Global support: expected 6, got {global_metrics['support']}"
    
    # 4. Verify global metrics via results_dict
    results_dict = metrics.results_dict
    
    # Verify all per-class metrics in dict
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
    
    # Verify global metrics in results_dict
    assert results_dict['precision/global'] == 6/(6+3)
    assert results_dict['recall/global'] == 1.0
    assert abs(results_dict['f1/global'] - expected_f1) < 1e-6
    assert results_dict['support/global'] == 6

    # Verify fitness metric
    assert metrics.fitness == results_dict['f1/global']

def test_map_calculation():
    """
    Teste abrangente para validar o cálculo do mAP (mean Average Precision).
    Testa diferentes cenários incluindo múltiplas classes, diferentes níveis de confiança,
    e diferentes thresholds de IoU.
    """
    
    names = {0: 'person', 1: 'car', 2: 'bike'}
    metrics = DetectionMetrics(names=names, iou_thr=0.5, conf_thr=0.1)
    
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
    
    gt_anns_2 = [
        {'category_id': 0, 'bbox': [200, 200, 25, 25]},
        {'category_id': 1, 'bbox': [300, 300, 40, 40]},
    ]
    
    pred_anns_2 = [
        {'category_id': 0, 'bbox': [202, 202, 23, 23], 'score': 0.88},
        {'category_id': 1, 'bbox': [305, 305, 35, 35], 'score': 0.82},
        {'category_id': 0, 'bbox': [400, 400, 20, 20], 'score': 0.75},
        {'category_id': 2, 'bbox': [500, 500, 10, 10], 'score': 0.70} 
    ]
    
    metrics.process_image(gt_anns_2, pred_anns_2)
    
    gt_anns_3 = [
        {'category_id': 1, 'bbox': [600, 600, 50, 50]},  # car
    ]
    
    pred_anns_3 = [
        {'category_id': 1, 'bbox': [620, 620, 30, 30], 'score': 0.60}, 
    ]
    
    metrics.process_image(gt_anns_3, pred_anns_3)
    
    gt_anns_4 = [
        {'category_id': 0, 'bbox': [700, 700, 30, 30]},
    ]
    
    pred_anns_4 = [
        {'category_id': 0, 'bbox': [702, 702, 28, 28], 'score': 0.95}, 
        {'category_id': 0, 'bbox': [705, 705, 25, 25], 'score': 0.85},
        {'category_id': 0, 'bbox': [710, 710, 20, 20], 'score': 0.75},
    ]
    
    metrics.process_image(gt_anns_4, pred_anns_4)
    result = metrics.compute_metrics()
    
    assert 'global' in result, "Métricas globais devem estar presentes"
    assert 'mAP50' in result['global'], "mAP@0.5 deve estar presente"
    assert 'mAP75' in result['global'], "mAP@0.75 deve estar presente"
    assert 'mAP' in result['global'], "mAP@0.5:0.95 deve estar presente"
    
    assert 0 <= result['global']['mAP50'] <= 1, f"mAP@0.5 fora do range: {result['global']['mAP50']}"
    assert 0 <= result['global']['mAP75'] <= 1, f"mAP@0.75 fora do range: {result['global']['mAP75']}"
    assert 0 <= result['global']['mAP'] <= 1, f"mAP fora do range: {result['global']['mAP']}"
    
    for class_id in names.keys():
        if class_id in result:
            assert 'AP50' in result[class_id], f"AP50 deve estar presente para classe {class_id}"
            assert 'AP75' in result[class_id], f"AP75 deve estar presente para classe {class_id}"
            assert 'AP' in result[class_id], f"AP deve estar presente para classe {class_id}"
            assert 0 <= result[class_id]['AP50'] <= 1, f"AP50 fora do range para classe {class_id}"
            assert 0 <= result[class_id]['AP75'] <= 1, f"AP75 fora do range para classe {class_id}"
            assert 0 <= result[class_id]['AP'] <= 1, f"AP fora do range para classe {class_id}"
    
    assert result['global']['mAP50'] >= result['global']['mAP75'], \
        "mAP@0.5 deve ser >= mAP@0.75"
    
    assert len(metrics.all_preds) > 0, "Deve haver predições armazenadas"
    assert len(metrics.all_gts) > 0, "Deve haver ground truths armazenados"


def test_map_edge_cases():
    """
    Testa casos extremos para o cálculo do mAP.
    """
    
    names = {0: 'person'}
    metrics = DetectionMetrics(names=names)
    
    gt_anns = [{'category_id': 0, 'bbox': [10, 10, 20, 20]}]
    pred_anns = []
    
    metrics.process_image(gt_anns, pred_anns)
    result = metrics.compute_metrics()
    
    if 'mAP50' in result['global']:
        assert result['global']['mAP50'] == 0.0, "mAP deve ser 0 quando não há predições"
    
    metrics.reset()
    gt_anns = []
    pred_anns = [{'category_id': 0, 'bbox': [10, 10, 20, 20], 'score': 0.8}]
    
    metrics.process_image(gt_anns, pred_anns)
    result = metrics.compute_metrics()
    
    if 'mAP50' in result['global']:
        assert result['global']['mAP50'] == 0.0, "mAP deve ser 0 quando não há ground truths"
    
    metrics = DetectionMetrics(names=names, conf_thr=0.9)
    
    gt_anns = [{'category_id': 0, 'bbox': [10, 10, 20, 20]}]
    pred_anns = [{'category_id': 0, 'bbox': [10, 10, 20, 20], 'score': 0.5}]  # Abaixo do threshold
    
    metrics.process_image(gt_anns, pred_anns)
    result = metrics.compute_metrics()
    
    if 'mAP50' in result['global']:
        assert result['global']['mAP50'] == 0.0, "mAP deve ser 0 quando predições são filtradas por confiança"


def test_map_consistency():
    """
    Testa a consistência do cálculo do mAP com diferentes ordens de processamento.
    """
    names = {0: 'person', 1: 'car'}
    metrics1 = DetectionMetrics(names=names)
    
    gt1 = [{'category_id': 0, 'bbox': [10, 10, 20, 20]}]
    pred1 = [{'category_id': 0, 'bbox': [12, 12, 18, 18], 'score': 0.9}]
    metrics1.process_image(gt1, pred1)
    
    gt2 = [{'category_id': 1, 'bbox': [50, 50, 30, 30]}]
    pred2 = [{'category_id': 1, 'bbox': [52, 52, 28, 28], 'score': 0.8}]
    metrics1.process_image(gt2, pred2)
    
    result1 = metrics1.compute_metrics()
    
    metrics2 = DetectionMetrics(names=names)
    metrics2.process_image(gt2, pred2)
    metrics2.process_image(gt1, pred1)
    
    result2 = metrics2.compute_metrics()
    
    if 'mAP50' in result1['global'] and 'mAP50' in result2['global']:
        assert abs(result1['global']['mAP50'] - result2['global']['mAP50']) < 1e-6, \
            "mAP deve ser consistente independente da ordem de processamento"
        
def test_different_maps():    
    names = {0: 'person', 1: 'car'}
    metrics = DetectionMetrics(names=names, iou_thr=0.5, conf_thr=0.1)
    def calculate_expected_iou(gt_bbox, pred_bbox):
        return DetectionMetrics.bbox_iou(gt_bbox, pred_bbox)    
    gt_anns_1 = [
        {'category_id': 0, 'bbox': [100, 100, 50, 50]},  # GT 1
        {'category_id': 0, 'bbox': [200, 200, 40, 40]},  # GT 2
        {'category_id': 0, 'bbox': [300, 300, 60, 60]},  # GT 3
        {'category_id': 0, 'bbox': [400, 400, 30, 30]},  # GT 4
    ]
    
    pred_anns_1 = [
        # Predição EXCELENTE (IoU ~0.84) - conta para todos os thresholds
        {'category_id': 0, 'bbox': [102, 102, 46, 46], 'score': 0.95},
        
        # Predição BOA (IoU ~0.64) - conta apenas para mAP@0.5
        {'category_id': 0, 'bbox': [205, 205, 30, 30], 'score': 0.90},
        
        # Predição MÉDIA (IoU ~0.56) - conta apenas para mAP@0.5
        {'category_id': 0, 'bbox': [310, 310, 40, 40], 'score': 0.85},
        
        # Predição RUIM (IoU ~0.25) - não conta para nenhum threshold
        {'category_id': 0, 'bbox': [420, 420, 20, 20], 'score': 0.80},
        
        # Falso Positivo
        {'category_id': 0, 'bbox': [500, 500, 25, 25], 'score': 0.75},
    ]
    
    # Calcular e mostrar IoUs esperados
    for i, pred in enumerate(pred_anns_1[:-1]):  # Excluir FP
        if i < len(gt_anns_1):
            iou = calculate_expected_iou(gt_anns_1[i]['bbox'], pred['bbox'])
    
    metrics.process_image(gt_anns_1, pred_anns_1)    
    gt_anns_2 = [
        {'category_id': 1, 'bbox': [50, 50, 80, 40]},   # GT 1
        {'category_id': 1, 'bbox': [150, 150, 70, 35]}, # GT 2
        {'category_id': 1, 'bbox': [250, 250, 60, 30]}, # GT 3
    ]
    
    pred_anns_2 = [
        # Predição EXCELENTE (IoU alto) - conta para todos
        {'category_id': 1, 'bbox': [52, 52, 76, 38], 'score': 0.92},
        
        # Predição MÉDIA (IoU ~0.6) - conta apenas para mAP@0.5
        {'category_id': 1, 'bbox': [160, 160, 50, 25], 'score': 0.88},
        
        # Miss - GT sem predição correspondente
        # (GT 3 ficará como False Negative)
        
        # Falso Positivo
        {'category_id': 1, 'bbox': [350, 350, 40, 20], 'score': 0.82},
    ]
    
    for i, pred in enumerate(pred_anns_2[:-1]):  # Excluir FP
        if i < len(gt_anns_2):
            iou = calculate_expected_iou(gt_anns_2[i]['bbox'], pred['bbox'])
    
    metrics.process_image(gt_anns_2, pred_anns_2)
        
    gt_anns_3 = [
        {'category_id': 0, 'bbox': [10, 10, 40, 40]},   # person
        {'category_id': 1, 'bbox': [100, 10, 50, 25]},  # car
    ]
    
    pred_anns_3 = [
        # Predição MUITO BOA para person (IoU ~0.78)
        {'category_id': 0, 'bbox': [12, 12, 36, 36], 'score': 0.93},
        
        # Predição MÉDIA para car (IoU ~0.58)
        {'category_id': 1, 'bbox': [105, 15, 40, 20], 'score': 0.87},
    ]
    
    for i, pred in enumerate(pred_anns_3):
        iou = calculate_expected_iou(gt_anns_3[i]['bbox'], pred['bbox'])
    
    metrics.process_image(gt_anns_3, pred_anns_3)    
    result = metrics.compute_metrics()
    
    # Mostrar métricas globais
    if 'global' in result and 'mAP50' in result['global']:
        assert result['global']['mAP50'] == np.float64(0.6750)
        assert result['global']['mAP75'] == np.float64(0.3250)
        assert result['global']['mAP'] == np.float64(0.31416666666666665)

def test_create_controlled_iou_example():
    names = {0: 'object'}
    metrics = DetectionMetrics(names=names, iou_thr=0.5, conf_thr=0.1)
    
    # Ground truth base
    gt_base = [100, 100, 100, 100]  # x, y, w, h
    
    # Criar predições com IoUs específicos
    test_cases = [
        # (deslocamento, tamanho, IoU_esperado, descrição)
        (0, 100, 1.00, "Perfeita"),           # Exata
        (10, 90, 0.81, "Excelente"),          # Pequeno deslocamento
        (20, 80, 0.64, "Boa"),                # Deslocamento médio  
        (25, 75, 0.56, "Média"),              # IoU próximo de 0.5
        (35, 65, 0.42, "Ruim"),               # Abaixo de 0.5
    ]
    
    gt_anns = []
    pred_anns = []
    
    for i, (offset, size, expected_iou, desc) in enumerate(test_cases):
        # Ground truth
        gt_bbox = [gt_base[0] + i*200, gt_base[1], gt_base[2], gt_base[3]]
        gt_anns.append({'category_id': 0, 'bbox': gt_bbox})
        
        # Predição correspondente
        pred_bbox = [gt_bbox[0] + offset, gt_bbox[1] + offset, size, size]
        pred_anns.append({
            'category_id': 0, 
            'bbox': pred_bbox, 
            'score': 0.9 - i*0.1  # Scores decrescentes
        })
        
        # Calcular IoU real
        real_iou = DetectionMetrics.bbox_iou(gt_bbox, pred_bbox)
    
    metrics.process_image(gt_anns, pred_anns)
    result = metrics.compute_metrics()
    assert 'global' in result, "Resultados globais devem estar presentes"
    assert result['global']['mAP50'] == np.float64(0.8)
    assert result['global']['mAP75'] == np.float64(0.4)
    assert result['global']['mAP'] == np.float64(0.44000000000000006)
