from utils import open_jsons, ajust_ground_truth
from object_detection_metrics import mIoU, coco_metric, precision_recall_f1_score, false_negatives, acuracy, balanced_accuracy
import os

coco_gt_path = r'.\tests\jsons\_annotations.coco.json'
coco_result_path = r'.\tests\jsons\tood_predicts_bbox.bbox.json'

gt_json, predictions_json = open_jsons([coco_gt_path, coco_result_path])
gt_json_ajusted = ajust_ground_truth(gt_json)

mIoU(gt_json_ajusted, predictions_json, thr_score= 0.4, verbose=True)
coco_metric(gt_json_ajusted, predictions_json, thr_score= 0.4, verbose=True)
precision_recall_f1_score(gt_json, predictions_json, thr_score= 0.4, verbose=True, iou_thr=0.5, skip_classes=[0], average='weighted')
false_negatives(gt_json, predictions_json, thr_score= 0.4, verbose=True, iou_thr=0.5, skip_classes=[0])
acuracy(gt_json, predictions_json, thr_score= 0.4, verbose=True, iou_thr=0.5, skip_classes=[0])
balanced_accuracy(gt_json, predictions_json, thr_score= 0.4, verbose=True, iou_thr=0.5, skip_classes=[0])
os.remove(gt_json_ajusted)