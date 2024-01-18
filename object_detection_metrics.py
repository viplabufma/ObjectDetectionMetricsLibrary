from utils import thr_score_on_prediction, get_prediction_for_image_id, calc_iou, generate_true_and_pred_vector, classification_metrics, open_jsons, ajust_ground_truth, draw_confusion_matrix
from utils import cls
from utils import calc_false_negatives_rate
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import os
from sklearn.metrics import accuracy_score, balanced_accuracy_score

def mIoU(gt_json_path, predictions_json, thr_score, verbose=True):
  predictions = thr_score_on_prediction(predictions_json, thr_score)
  gt_coco = COCO(gt_json_path)
  ious = []
  for gt_ann in gt_coco.dataset['annotations']:
      pred_anns = get_prediction_for_image_id(gt_ann, predictions)
      ious.extend(calc_iou(gt_ann, pred_anns))
  if verbose:
    print(f'\nmIoU: {np.mean(ious)}')
  return np.mean(ious)

# TODO: 
# Verbose: Não esta funionando corretamente
def coco_metric(gt_json, predictions_json, thr_score= 0.0, verbose=True):
  predictions = thr_score_on_prediction(predictions_json, thr_score)
  gt_coco = COCO(gt_json)
  pred_coco = gt_coco.loadRes(predictions)
  coco_eval = COCOeval(gt_coco, pred_coco, 'bbox')
  coco_eval.evaluate()
  coco_eval.accumulate()
  coco_eval.summarize()
  if not verbose:
    cls()
  return coco_eval.stats[0]

def precision_recall_f1_score(gt_json, predictions_json, thr_score= 0.0, verbose=True, iou_thr=0.5, skip_classes=[0], average='weighted'):
  y_true, y_pred, classes = generate_true_and_pred_vector(gt_json, predictions_json, thr_score, iou_thr, skip_classes=skip_classes, verbose=False)
  return classification_metrics(y_pred, y_true, classes, verbose=verbose, average=average)


def false_negatives(gt_json, predictions_json, thr_score= 0.0, verbose=True, iou_thr=0.5, skip_classes=[0]):
  y_true, y_pred,_  = generate_true_and_pred_vector(gt_json, predictions_json, thr_score, iou_thr, skip_classes=skip_classes, verbose=False)
  conf_matrix = draw_confusion_matrix(y_true, y_pred, verbose=False)
  num_classes = len(set(np.concatenate((y_pred, y_true))))
  return calc_false_negatives_rate(num_classes, conf_matrix, verbose=False)

def acuracy(gt_json, predictions_json, thr_score= 0.0, verbose=True, iou_thr=0.5, skip_classes=[0]):
  y_true, y_pred,_  = generate_true_and_pred_vector(gt_json, predictions_json, thr_score, iou_thr, skip_classes=skip_classes, verbose=False)
  acc = accuracy_score(y_true, y_pred)
  if verbose:
    print(f'\nAcuracy: {acc}')
  return acc

def balanced_accuracy(gt_json, predictions_json, thr_score= 0.0, verbose=True, iou_thr=0.5, skip_classes=[0]):
  y_true, y_pred,_  = generate_true_and_pred_vector(gt_json, predictions_json, thr_score, iou_thr, skip_classes=skip_classes, verbose=False)
  acc = balanced_accuracy_score(y_true, y_pred)
  if verbose:
    print(f'\nBalanced Acuracy: {acc}')
  return acc

