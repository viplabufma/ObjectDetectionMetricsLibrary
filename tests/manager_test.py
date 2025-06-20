'''
Author: Matheus Levy
Organization: Viplab - UFMA
GitHub: https://github.com/viplabufma/MatheusLevy_mestrado
'''
from DetectionMetricManager import DetectionMetricsManager, export_metrics, save_confusion_matrix
import numpy as np
import json

def test_precision_simple():
    gt_json_path = "/home/thebig/Documentos/MatheusLevy_mestrado/tests/jsons/simple/gt_coco.json"
    predictions_json_path = "/home/thebig/Documentos/MatheusLevy_mestrado/tests/jsons/simple/predictions_coco.json"
    manager = DetectionMetricsManager(gt_path=gt_json_path,result_path=predictions_json_path)
    manager.load_data()
    metrics = manager.calculate_metrics()
    save_confusion_matrix(metrics['confusion_matrix_multiclass'], manager.labels,'confusion_matrix.png', background_class=True)
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
    gt_json_path = "/home/thebig/Documentos/MatheusLevy_mestrado/tests/jsons/medium/gt_coco.json"
    predictions_json_path = "/home/thebig/Documentos/MatheusLevy_mestrado/tests/jsons/medium/predictions_coco.json"
    manager = DetectionMetricsManager(gt_path=gt_json_path,result_path=predictions_json_path)
    manager.load_data()
    metrics = manager.calculate_metrics()
    print(metrics)
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

def test_expor_json():
    gt_json_path = "/home/thebig/Documentos/MatheusLevy_mestrado/tests/jsons/simple/gt_coco.json"
    predictions_json_path = "/home/thebig/Documentos/MatheusLevy_mestrado/tests/jsons/simple/predictions_coco.json"
    manager = DetectionMetricsManager(gt_path=gt_json_path,result_path=predictions_json_path)
    manager.load_data()
    metrics = manager.calculate_metrics()
    export_metrics(metrics, manager.labels)
    with open('metrics.json', 'r') as f:
        data = json.load(f)
    assert data['person']['precision'] == 0.6666666666666666
    assert data['person']['recall'] == 0.6666666666666666
    assert data['person']['f1'] == 0.6666666666666666
    assert data['person']['support'] == 3
    assert data['global']['precision'] == 0.6666666666666666
    assert data['global']['recall'] == 0.6666666666666666
    assert data['global']['f1'] == 0.6666666666666666
    assert data['global']['support'] == 3
    assert data['global']['mAP50'] == 0.16831683168316833
    assert data['global']['mAP75'] == 0.16831683168316833
    assert data['global']['mAP'] == 0.16831683168316833
