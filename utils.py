from sklearn.metrics import precision_recall_fscore_support
import seaborn as sns;
from sklearn.metrics import confusion_matrix
import json
from tqdm import tqdm
import numpy as np
import os

def calc_false_negatives_rate(num_classes, conf_matrix, verbose=True):
  _output = '\n'
  fn_rate_list = []
  for i in range(1, num_classes):
    false_negatives = conf_matrix[i, 0]
    total_true_positives = np.sum(conf_matrix[i, 1:])
    fn_rate = false_negatives / total_true_positives if total_true_positives != 0 else 0
    fn_rate_list.append(fn_rate)
    _output+= f"Taxa de Falsos Negativos para a Classe {i}: {fn_rate*100}\n"
  if verbose:
    print(_output)
  return fn_rate_list

def draw_confusion_matrix(true,preds, verbose=True):
  num_classes = len(set(np.concatenate((preds, true))))
  conf_matx = confusion_matrix(true, preds)
  calc_false_negatives_rate(num_classes, conf_matx, verbose=verbose)
  if verbose:
    sns.heatmap(conf_matx, annot=True, annot_kws={"size": 12}, fmt='g', cbar=False, cmap="viridis")
  return conf_matx

def calculate_f1_score_per_class(vetor_groundtruth, vetor_predicoes, classes, average= 'weighted'):
    precision, recall, f1_score, _ = precision_recall_fscore_support(vetor_groundtruth, vetor_predicoes, labels=classes, average=None)
    results_per_class = {}
    for i, class_id in enumerate(classes):
        results_per_class[class_id] = {
            'Precision': precision[i],
            'Recall': recall[i],
            'F1-score': f1_score[i]
        }
    return results_per_class

def classification_metrics(predito_vetor, groundtruth_vetor, classes, average='weighted', verbose=True):
  _output = ''
  f1_score_per_class = calculate_f1_score_per_class(groundtruth_vetor, predito_vetor, classes, average= average)
  precision_mean, recall_mean, f1_score_mean  = 0, 0, 0

  for class_id, metrics in f1_score_per_class.items():
      _output += f'Classe {class_id}: Precision={metrics["Precision"]}, Recall={metrics["Recall"]}, F1-score={metrics["F1-score"]}\n'
      precision_mean += metrics['Precision']
      recall_mean += metrics['Recall']
      f1_score_mean += metrics['F1-score']
  
  if verbose:
    print(_output)
    print(f'mean Precision: {precision_mean/len(classes)}')
    print(f'mean Recall: {recall_mean/len(classes)}')
    print(f'mean F1-Score: {f1_score_mean/len(classes)}')

  draw_confusion_matrix(groundtruth_vetor,predito_vetor, verbose=verbose)
  return f1_score_mean/len(classes), precision_mean/len(classes), recall_mean/len(classes)

def add_images_to_json(images_anns, dict):
  dict['images'] = list()
  for img in images_anns:
    dict['images'].append({
        'id': img['id'],
        'width': img['width'],
        'height': img['height'],
        'file_name': img['file_name']
    })
  return dict

def add_annotations_to_json(annotations, dict):
  dict['annotations'] = list()
  for ann in annotations:
    dict['annotations'].append({
        'id': ann['id'],
        'image_id': ann['image_id'],
        'category_id': ann['category_id'],
        'bbox': ann['bbox'],
        'iscrowd':0,
        'area':ann['area']
    })
  return dict

def add_categories_to_json(categories, dict):
  dict['categories'] = list()
  for cat in categories:
    dict['categories'].append({
        'id': cat['id'],
        'name': cat['name']
    })
  return dict

def save_json(dict, caminho_arquivo_json):
  with open(caminho_arquivo_json, 'w') as arquivo:
      json.dump(dict, arquivo, indent=4)

def ajust_ground_truth(gt, caminho_arquivo_json = "ajusted_gt.json"):
  gt_ajustado = dict()
  gt_ajustado = add_images_to_json(gt['images'], gt_ajustado)
  gt_ajustado = add_annotations_to_json(gt['annotations'], gt_ajustado)
  gt_ajustado = add_categories_to_json(gt['categories'], gt_ajustado)
  save_json(gt_ajustado, caminho_arquivo_json)
  return caminho_arquivo_json
   
def cls():
    os.system('cls' if os.name=='nt' else 'clear')
  
def bbox_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)

    inter_area = max(inter_x2 - inter_x1, 0) * max(inter_y2 - inter_y1, 0)

    bbox1_area = w1 * h1
    bbox2_area = w2 * h2

    union_area = bbox1_area + bbox2_area - inter_area
    iou = inter_area / union_area
    return iou

def get_prediction_for_image_id(gt_annotation, predictions):
  image_id = gt_annotation['image_id']
  return [ann for ann in predictions if ann['image_id'] == image_id]

def calc_iou(gt_ann, pred_anns):
  ious = []
  gt_bbox = gt_ann['bbox']
  for pred_ann in pred_anns:
    pred_bbox = pred_ann['bbox']
    iou = bbox_iou(gt_bbox, pred_bbox)
    ious.append(iou)
  return ious

def no_more_predictions(gt_anns):
  preds, trues = [], []
  for ann_left in gt_anns:
    trues.append(ann_left["category_id"])
    preds.append(-1)
  return trues, preds

def match_best_class_and_best_iou(list_of_predictions, ann_true):
  best_iou = 0
  best_index = -1
  bbox_true = ann_true["bbox"]
  for i in range(len(list_of_predictions)):
    ann_pred = list_of_predictions[i]
    bbox_pred = ann_pred["bbox"]
    iou = bbox_iou(bbox_true, bbox_pred)
    if iou > best_iou and ann_pred["category_id"] == ann_true["category_id"]:
      best_iou = iou
      best_index = i
  return best_index, best_iou

def match_best_only_iou(list_of_predictions, ann_true):
  best_iou = 0
  best_index = -1
  bbox_true = ann_true["bbox"]
  for i in range(len(list_of_predictions)):
    ann_pred = list_of_predictions[i]
    bbox_pred = ann_pred["bbox"]
    iou = bbox_iou(bbox_true, bbox_pred)
    if iou > best_iou:
      best_iou = iou
      best_index = i
  return best_index, best_iou

def find_best_match(list_of_predictions, ann_true):
  best_index, best_iou = match_best_class_and_best_iou(list_of_predictions, ann_true)
  if best_index != -1:
    return best_index, best_iou
  else:
    best_index, best_iou  = match_best_only_iou(list_of_predictions, ann_true)
    return best_index, best_iou

def isEmpty(list):
  return len(list) == 0

def get_classes_from_pred_and_groundtruth(gt_anns, pred_anns):
  pred_class = pred_anns['category_id']
  true_class = gt_anns['category_id']
  return pred_class, true_class

def generate_preds_true_vector_from_anns(gt_anns, pred_anns, iou_thr):
  y_pred, y_true  = [], []

  for index, ann_gt in enumerate(gt_anns):
    if isEmpty(pred_anns): # False Negative
      trues, preds = no_more_predictions(gt_anns)
      y_pred.extend(preds)
      y_true.extend(trues)
      gt_anns = []
      break

    index_best_pred, best_iou = find_best_match(pred_anns, ann_gt)
    if best_iou > iou_thr:
      best_pred = pred_anns[index_best_pred]
      class_pred, class_true = get_classes_from_pred_and_groundtruth(ann_gt, best_pred)
      if class_pred == class_true: # True Positive
        y_pred.append(class_pred)
        y_true.append(class_true)
        gt_anns.pop(index)
        pred_anns.pop(index_best_pred)
      else: # False Postive: IoU >= thr, porém a classe está errada.
        y_pred.append(class_pred)
        y_true.append(class_true)
        gt_anns.pop(index)
        pred_anns.pop(index_best_pred)
    
  for pred in pred_anns: # False Positive: A predição não possuim IoU com o groundtruth. Todos elementos do groundtruth estão marcadas mas existem elementos não marcados na lista de predições.
    y_pred.append(pred["category_id"])
    y_true.append(-1)
  
  for gt in gt_anns: # False Negative: Todas minhas predições ja foram marcadas (associadas a uma bounding box do gt)
    y_pred.append(-1)
    y_true.append(gt["category_id"])

  return y_true, y_pred

def get_all_bboxes_from_image_id(gt_coco, id):
   return [ann for ann in gt_coco['annotations'] if ann['image_id'] == id]

def thr_score_on_prediction(predictions_json, score):
  return [pred for pred in predictions_json if pred['score'] >= score]

def get_all_ids_from_coco_gt_json(gt_json):
  return [imagem["id"] for imagem in gt_json["images"]]

def get_all_predictions_from_image_id(predictions_json, id):
  return [pred for pred in predictions_json if pred["image_id"] == id]

def get_classes_id_from_coco_gt_json(gt_json, skip_classes = []):
  return [c["id"] for c in gt_json["categories"] if c['id'] not in skip_classes]
  
def generate_true_and_pred_vector(gt_json, predictions_json, thr_score, iou_thr, skip_classes = [], verbose= True):
  predictions_json = thr_score_on_prediction(predictions_json, thr_score)
  ids_das_imagens = get_all_ids_from_coco_gt_json(gt_json)
  y_true_full, y_pred_full = [], []
  classes = get_classes_id_from_coco_gt_json(gt_json, skip_classes)
  
  for id in tqdm(ids_das_imagens, disable= not verbose):
    preds = get_all_predictions_from_image_id(predictions_json, id)
    gts = get_all_bboxes_from_image_id(gt_json, id)
    y_true, y_pred = generate_preds_true_vector_from_anns(gts, preds, iou_thr)
    y_true_full.extend(y_true)
    y_pred_full.extend(y_pred)
  return y_true_full, y_pred_full, classes

def open_jsons(paths):
    jsons = []
    for path in paths:
        with open(path, 'r') as f:
            jsons.append(json.load(f))
    return jsons
