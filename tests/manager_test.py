from DetectionMetricManager import DetectionMetricsManager
import numpy as np

def test_precision_simple():
    gt_json_path = "/home/thebig/Documentos/MatheusLevy_mestrado/tests/jsons/simple/gt_coco.json"
    predictions_json_path = "/home/thebig/Documentos/MatheusLevy_mestrado/tests/jsons/simple/predictions_coco.json"
    manager = DetectionMetricsManager(gt_path=gt_json_path,result_path=predictions_json_path)
    manager.load_data()
    metrics = manager.calculate_metrics()
    assert metrics[0]['precision'] == np.float64(0.6666666666666666)
    assert metrics[0]['recall'] == np.float64(0.6666666666666666)
    assert metrics[0]['f1'] == np.float64(0.6666666666666666)
    assert metrics[0]['support'] == np.int64(3)
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
    assert metrics[0]['precision'] == np.float64(1.0)
    assert metrics[0]['recall'] == np.float64(0.3333333333333333)
    assert metrics[0]['f1'] == np.float64(0.5)
    assert metrics[0]['support'] == np.int64(3)
    assert metrics[1]['precision'] == np.float64(0.5)
    assert metrics[1]['recall'] == np.float64(0.5)
    assert metrics[1]['f1'] == np.float64(0.5)
    assert metrics[1]['support'] == np.int64(2)
    assert metrics[2]['precision'] == np.float64(1.0)
    assert metrics[2]['recall'] == np.float64(0.5)
    assert metrics[2]['f1'] == np.float64(0.6666666666666666)
    assert metrics[2]['support'] == np.int64(2)
    assert metrics['global']['precision'] == np.float64(0.75)
    assert metrics['global']['recall'] ==  np.float64(0.42857142857142855)
    assert metrics['global']['f1'] == np.float64(0.5454545454545454)
    assert metrics['global']['support'] == np.int64(7)
    assert metrics['global']['mAP50'] == np.float64(0.3366336633663366)
    assert metrics['global']['mAP75'] == np.float64(0.3366336633663366)
    assert metrics['global']['mAP'] == np.float64(0.33663366336633654)


