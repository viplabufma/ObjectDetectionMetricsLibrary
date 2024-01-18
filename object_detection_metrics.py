from utils import thr_score_on_prediction, get_prediction_for_image_id, calc_iou, generate_true_and_pred_vector, classification_metrics, open_jsons, ajust_ground_truth, draw_confusion_matrix
from utils import cls
from utils import calc_false_negatives_rate
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

def mIoU(gt_json_path: str, predictions_json:str, thr_score:float, verbose:bool=True):
  """
  Calculate the Mean intersection over union (mIoU) metric for a given ground truth and predictions.

  Args:
  gt_json_path (str): Path to ground truth json in coco format.
  predicitons_json (str): Path to result jsons in coco format.
  thr_score (float): Threshold to filter predictions by score.
  verbose (bool): If True, print the mIoU value.

  Returns:
  float: mIoU value.

  Exemplo:
  >>> mIoU('ground_truth.json', 'predictons_result.json', 0.3, verbose=True)
  0.98332
  """
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
  """
  Calculate the mAP metric for a given ground truth and predictions.

  Args:
  gt_json (dict): Dictionary with ground truth in coco format.
  predicitons_json (dict): Dictionary with result in coco format.
  thr_score (float): Threshold to filter predictions by score.
  verbose (bool): If True, print the mIoU value.

  Returns:
  float: mAP value.

  Exemplo:
  >>> coco_metric(ground_truth_dict, predictions_dict, 0.3, verbose=True)
  0.973
  """
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
  """
  Calculate precision, recall and f1-score metrics for a given ground truth and predictions.

  Args:
  gt_json (dict): Dictionary with ground truth in coco format.
  predicitons_json (dict): Dictionary with result in coco format.
  thr_score (float): Threshold to filter predictions by score.
  verbose (bool): If True, print the mIoU value.
  iou_thr (float): Threshold to consider a prediction and groundthruth as a match (or refering to the same object).
  skip_classes (list): List of classes to skip in the calculation.
  average (str): Type of average to calculate the metrics. Can be 'weighted', 'macro' or 'micro'.

  Returns:
  Tuple: (f1-score, precision, recall)

  Exemplo:
  >>> precision_recall_f1_score(ground_truth_dict, predictions_dict, 0.3, verbose=True)
  0.973, 0.973, 0.973
  """
  y_true, y_pred, classes = generate_true_and_pred_vector(gt_json, predictions_json, thr_score, iou_thr, skip_classes=skip_classes, verbose=False)
  return classification_metrics(y_pred, y_true, classes, verbose=verbose, average=average)


def false_negatives(gt_json, predictions_json, thr_score= 0.0, verbose=True, iou_thr=0.5, skip_classes=[0]):
  """
  Calculate the false negatives rate for a given ground truth and predictions.

  Args:
  gt_json (dict): Dictionary with ground truth in coco format.
  predicitons_json (dict): Dictionary with result in coco format.
  thr_score (float): Threshold to filter predictions by score.
  verbose (bool): If True, print the mIoU value.
  iou_thr (float): Threshold to consider a prediction and groundthruth as a match (or refering to the same object).
  skip_classes (list): List of classes to skip in the calculation.
  average (str): Type of average to calculate the metrics. Can be 'weighted', 'macro' or 'micro'.

  Returns:
  array: Array with false negatives rate for each class.

  Exemplo:
  >>> false_negatives(ground_truth_dict, predictions_dict, 0.3, verbose=True)
  [0.05, 0.073, 0.2, 0.0, 0.2, 0.1, 0.345]
  """
  y_true, y_pred,_  = generate_true_and_pred_vector(gt_json, predictions_json, thr_score, iou_thr, skip_classes=skip_classes, verbose=False)
  conf_matrix = draw_confusion_matrix(y_true, y_pred, verbose=False)
  num_classes = len(set(np.concatenate((y_pred, y_true))))
  return calc_false_negatives_rate(num_classes, conf_matrix, verbose=verbose)

def accuracy(gt_json, predictions_json, thr_score= 0.0, verbose=True, iou_thr=0.5, skip_classes=[0]):
  """
  Calculate the accuracy for a given ground truth and predictions.

  Args:
  gt_json (dict): Dictionary with ground truth in coco format.
  predicitons_json (dict): Dictionary with result in coco format.
  thr_score (float): Threshold to filter predictions by score.
  verbose (bool): If True, print the mIoU value.
  iou_thr (float): Threshold to consider a prediction and groundthruth as a match (or refering to the same object).
  skip_classes (list): List of classes to skip in the calculation.
  average (str): Type of average to calculate the metrics. Can be 'weighted', 'macro' or 'micro'.

  Returns:
  float: Accuracy value.

  Exemplo:
  >>> accuracy(ground_truth_dict, predictions_dict, 0.3, verbose=True)
  0.973
  """
  y_true, y_pred,_  = generate_true_and_pred_vector(gt_json, predictions_json, thr_score, iou_thr, skip_classes=skip_classes, verbose=False)
  acc = accuracy_score(y_true, y_pred)
  if verbose:
    print(f'\nAcuracy: {acc}')
  return acc

def balanced_accuracy(gt_json, predictions_json, thr_score= 0.0, verbose=True, iou_thr=0.5, skip_classes=[0]):
  """
  Calculate the balanced accuracy for a given ground truth and predictions.

  Args:
  gt_json (dict): Dictionary with ground truth in coco format.
  predicitons_json (dict): Dictionary with result in coco format.
  thr_score (float): Threshold to filter predictions by score.
  verbose (bool): If True, print the mIoU value.
  iou_thr (float): Threshold to consider a prediction and groundthruth as a match (or refering to the same object).
  skip_classes (list): List of classes to skip in the calculation.
  average (str): Type of average to calculate the metrics. Can be 'weighted', 'macro' or 'micro'.

  Returns:
  float: Balanced accuracy value.

  Exemplo:
  >>> balanced_accuracy(ground_truth_dict, predictions_dict, 0.3, verbose=True)
  0.8345
  """
  y_true, y_pred,_  = generate_true_and_pred_vector(gt_json, predictions_json, thr_score, iou_thr, skip_classes=skip_classes, verbose=False)
  acc = balanced_accuracy_score(y_true, y_pred)
  if verbose:
    print(f'\nBalanced Acuracy: {acc}')
  return acc

