'''
Author: Matheus Levy
Organization: Viplab - UFMA
GitHub: https://github.com/viplabufma/MatheusLevy_mestrado
'''
from DetectionMetricManager import DetectionMetricsManager, export_metrics, save_confusion_matrix, plot_pr_curves
import numpy as np
import json
import pytest

def test_precision_simple():
    gt_json_path = "/home/thebig/Documentos/MatheusLevy_mestrado/tests/jsons/simple/gt_coco.json"
    predictions_json_path = "/home/thebig/Documentos/MatheusLevy_mestrado/tests/jsons/simple/predictions_coco.json"
    manager = DetectionMetricsManager(gt_path=gt_json_path,result_path=predictions_json_path)
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
    metrics = manager.calculate_metrics()
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
    gt_json_path = "/home/thebig/Documentos/MatheusLevy_mestrado/tests/jsons/real_case/_annotations.coco.json"
    predictions_json_path = "/home/thebig/Documentos/MatheusLevy_mestrado/tests/jsons/real_case/tood_predicts_bbox.bbox.json"
    manager = DetectionMetricsManager(gt_path=gt_json_path, result_path=predictions_json_path)
    metrics = manager.calculate_metrics(exclude_class=[0])
    export_metrics(metrics)
    plot_pr_curves(metrics['pr_curves'], output_path='./pr.png', show= False)
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
    
    # Capillaria philippinensis
    assert data['Capillaria philippinensis']['precision'] == pytest.approx(0.8866995073891626)
    assert data['Capillaria philippinensis']['recall'] == pytest.approx(0.8866995073891626)
    assert data['Capillaria philippinensis']['f1'] == pytest.approx(0.8866995073891626)
    assert data['Capillaria philippinensis']['support'] == 203
    assert data['Capillaria philippinensis']['tp'] == 180
    assert data['Capillaria philippinensis']['fp'] == 23
    assert data['Capillaria philippinensis']['fn'] == 23
    assert data['Capillaria philippinensis']['iou'] == pytest.approx(0.923344075679779)
    assert data['Capillaria philippinensis']['agg_iou'] == pytest.approx(0.9216952919960022)
    
    # Enterobius vermicularis
    assert data['Enterobius vermicularis']['precision'] == pytest.approx(0.9888268156424581)
    assert data['Enterobius vermicularis']['recall'] == pytest.approx(0.885)
    assert data['Enterobius vermicularis']['f1'] == pytest.approx(0.9340369393139841)
    assert data['Enterobius vermicularis']['support'] == 200
    assert data['Enterobius vermicularis']['tp'] == 177
    assert data['Enterobius vermicularis']['fp'] == 2
    assert data['Enterobius vermicularis']['fn'] == 23
    assert data['Enterobius vermicularis']['iou'] == pytest.approx(0.9346954226493835)
    assert data['Enterobius vermicularis']['agg_iou'] == pytest.approx(0.9344586133956909)
    
    # Fasciolopsis buski
    assert data['Fasciolopsis buski']['precision'] == pytest.approx(0.9846938775510204)
    assert data['Fasciolopsis buski']['recall'] == pytest.approx(0.9698492462311558)
    assert data['Fasciolopsis buski']['f1'] == pytest.approx(0.9772151898734177)
    assert data['Fasciolopsis buski']['support'] == 199
    assert data['Fasciolopsis buski']['tp'] == 193
    assert data['Fasciolopsis buski']['fp'] == 3
    assert data['Fasciolopsis buski']['fn'] == 6
    assert data['Fasciolopsis buski']['iou'] == pytest.approx(0.9501147270202637)
    assert data['Fasciolopsis buski']['agg_iou'] == pytest.approx(0.9486567378044128)
    
    # Hookworm egg
    assert data['Hookworm egg']['precision'] == pytest.approx(0.9902439024390244)
    assert data['Hookworm egg']['recall'] == pytest.approx(0.9950980392156863)
    assert data['Hookworm egg']['f1'] == pytest.approx(0.9926650366748165)
    assert data['Hookworm egg']['support'] == 204
    assert data['Hookworm egg']['tp'] == 203
    assert data['Hookworm egg']['fp'] == 2
    assert data['Hookworm egg']['fn'] == 1
    assert data['Hookworm egg']['iou'] == pytest.approx(0.9314258694648743)
    assert data['Hookworm egg']['agg_iou'] == pytest.approx(0.9329558610916138)
    
    # Hymenolepis diminuta
    assert data['Hymenolepis diminuta']['precision'] == pytest.approx(0.8032128514056225)
    assert data['Hymenolepis diminuta']['recall'] == pytest.approx(1.0)
    assert data['Hymenolepis diminuta']['f1'] == pytest.approx(0.89086859688196)
    assert data['Hymenolepis diminuta']['support'] == 200
    assert data['Hymenolepis diminuta']['tp'] == 200
    assert data['Hymenolepis diminuta']['fp'] == 49
    assert data['Hymenolepis diminuta']['fn'] == 0
    assert data['Hymenolepis diminuta']['iou'] == pytest.approx(0.9692001342773438)
    assert data['Hymenolepis diminuta']['agg_iou'] == pytest.approx(0.9683241248130798)
    
    # Hymenolepis nana
    assert data['Hymenolepis nana']['precision'] == pytest.approx(0.8936170212765957)
    assert data['Hymenolepis nana']['recall'] == pytest.approx(0.8316831683168316)
    assert data['Hymenolepis nana']['f1'] == pytest.approx(0.8615384615384615)
    assert data['Hymenolepis nana']['support'] == 202
    assert data['Hymenolepis nana']['tp'] == 168
    assert data['Hymenolepis nana']['fp'] == 20
    assert data['Hymenolepis nana']['fn'] == 34
    assert data['Hymenolepis nana']['iou'] == pytest.approx(0.9314448237419128)
    assert data['Hymenolepis nana']['agg_iou'] == pytest.approx(0.92930668592453)
    
    # Opisthorchis viverrine
    assert data['Opisthorchis viverrine']['precision'] == pytest.approx(0.9946236559139785)
    assert data['Opisthorchis viverrine']['recall'] == pytest.approx(0.925)
    assert data['Opisthorchis viverrine']['f1'] == pytest.approx(0.9585492227979274)
    assert data['Opisthorchis viverrine']['support'] == 200
    assert data['Opisthorchis viverrine']['tp'] == 185
    assert data['Opisthorchis viverrine']['fp'] == 1
    assert data['Opisthorchis viverrine']['fn'] == 15
    assert data['Opisthorchis viverrine']['iou'] == pytest.approx(0.9215582013130188)
    assert data['Opisthorchis viverrine']['agg_iou'] == pytest.approx(0.917792022228241)
    
    # Paragonimus spp
    assert data['Paragonimus spp']['precision'] == pytest.approx(0.8918918918918919)
    assert data['Paragonimus spp']['recall'] == pytest.approx(0.9473684210526315)
    assert data['Paragonimus spp']['f1'] == pytest.approx(0.9187935034802783)
    assert data['Paragonimus spp']['support'] == 209
    assert data['Paragonimus spp']['tp'] == 198
    assert data['Paragonimus spp']['fp'] == 24
    assert data['Paragonimus spp']['fn'] == 11
    assert data['Paragonimus spp']['iou'] == pytest.approx(0.957493782043457)
    assert data['Paragonimus spp']['agg_iou'] == pytest.approx(0.9559613466262817)
    
    # Taenia spp egg
    assert data['Taenia spp egg']['precision'] == pytest.approx(1.0)
    assert data['Taenia spp egg']['recall'] == pytest.approx(0.9411764705882353)
    assert data['Taenia spp egg']['f1'] == pytest.approx(0.9696969696969697)
    assert data['Taenia spp egg']['support'] == 204
    assert data['Taenia spp egg']['tp'] == 192
    assert data['Taenia spp egg']['fp'] == 0
    assert data['Taenia spp egg']['fn'] == 12
    assert data['Taenia spp egg']['iou'] == pytest.approx(0.9600203037261963)
    assert data['Taenia spp egg']['agg_iou'] == pytest.approx(0.9605120420455933)
    
    # Trichuris trichiura
    assert data['Trichuris trichiura']['precision'] == pytest.approx(1.0)
    assert data['Trichuris trichiura']['recall'] == pytest.approx(0.725)
    assert data['Trichuris trichiura']['f1'] == pytest.approx(0.8405797101449275)
    assert data['Trichuris trichiura']['support'] == 200
    assert data['Trichuris trichiura']['tp'] == 145
    assert data['Trichuris trichiura']['fp'] == 0
    assert data['Trichuris trichiura']['fn'] == 55
    assert data['Trichuris trichiura']['iou'] == pytest.approx(0.9440402388572693)
    assert data['Trichuris trichiura']['agg_iou'] == pytest.approx(0.9440361261367798)