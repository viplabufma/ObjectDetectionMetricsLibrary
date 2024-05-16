from utils import open_jsons, ajust_ground_truth
from object_detection_metrics import mIoU, coco_metric, precision_recall_f1_score, false_negatives, accuracy, balanced_accuracy
import os

coco_gt_path = r'C:\Users\levyb\Documents\MatheusLevy_mestrado\jsons\One Class\gt\test\_annotations.coco.json'
coco_result_path = r'C:\Users\levyb\Documents\MatheusLevy_mestrado\jsons\ensembles\ensemble_test_cascade_tood_modelo_proposto.json'

gt_json, predictions_json = open_jsons([coco_gt_path, coco_result_path])
gt_json_ajusted = ajust_ground_truth(gt_json)

thr_score = 0.4
mIoU(gt_json_ajusted, predictions_json, thr_score= thr_score, verbose=True)
coco_metric(gt_json_ajusted, predictions_json, thr_score= thr_score, verbose=True)
precision_recall_f1_score(gt_json, predictions_json, thr_score= thr_score, verbose=True, iou_thr=0.5, skip_classes=[0], average='weighted')
false_negatives(gt_json, predictions_json, thr_score= thr_score, verbose=True, iou_thr=0.5, skip_classes=[0])
accuracy(gt_json, predictions_json, thr_score= thr_score, verbose=True, iou_thr=0.5, skip_classes=[0])
balanced_accuracy(gt_json, predictions_json, thr_score= thr_score, verbose=True, iou_thr=0.5, skip_classes=[0])