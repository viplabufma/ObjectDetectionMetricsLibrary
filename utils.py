from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.metrics import precision_recall_fscore_support
import sklearn
import seaborn as sns;
from sklearn.metrics import cohen_kappa_score, roc_auc_score, roc_curve,accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
from tqdm import tqdm
import numpy as np
import os

def classification_metrics(predito_vetor, groundtruth_vetor, classes, average='weighted'):
  def draw_confusion_matrix(true,preds):
    num_classes = len(set(np.concatenate((preds, true))))
    conf_matx = confusion_matrix(true, preds)
    # Calcula a taxa de falsos negativos para cada classe
    for i in range(1, num_classes):
      false_negatives = conf_matx[i, 0]  # Falsos negativos para a classe i
      total_true_positives = np.sum(conf_matx[i, 1:])  # Verdadeiros positivos para a classe i

      fn_rate = false_negatives / total_true_positives if total_true_positives != 0 else 0
      print(f"Taxa de Falsos Negativos para a Classe {i}: {fn_rate*100}")

    sns.heatmap(conf_matx, annot=True,annot_kws={"size": 12},fmt='g', cbar=False, cmap="viridis")

  def calculate_f1_score_per_class(vetor_groundtruth, vetor_predicoes, classes, average= 'weighted'):
      # Calcular precision, recall e f1-score para cada classe
      precision, recall, f1_score, _ = precision_recall_fscore_support(vetor_groundtruth, vetor_predicoes, labels=classes, average=None)

      # Criar um dicionário para armazenar os resultados por classe
      results_per_class = {}
      for i, class_id in enumerate(classes):
          results_per_class[class_id] = {
              'Precision': precision[i],
              'Recall': recall[i],
              'F1-score': f1_score[i]
          }

      return results_per_class

  f1_score_per_class = calculate_f1_score_per_class(groundtruth_vetor, predito_vetor, classes, average= average)

  precision_mean = 0
  recall_mean = 0
  f1_score_mean = 0
  # Imprimir os resultados
  for class_id, metrics in f1_score_per_class.items():
      print(f'Classe {class_id}: Precision={metrics["Precision"]}, Recall={metrics["Recall"]}, F1-score={metrics["F1-score"]}')
      precision_mean += metrics['Precision']
      recall_mean += metrics['Recall']
      f1_score_mean += metrics['F1-score']

  print(f'mean Precision: {precision_mean/len(classes)}')
  print(f'mean Recall: {recall_mean/len(classes)}')
  print(f'mean F1-Score: {f1_score_mean/len(classes)}')

  draw_confusion_matrix(groundtruth_vetor,predito_vetor)

  return f1_score_mean/len(classes)

def ajust_ground_truth(gt, caminho_arquivo_json = "ajusted_gt.json"):
  # Ajustar GT
  gt_ajustado = dict()
  gt_ajustado['images'] = list()

  for img in gt['images']:
    img_id = img['id']
    width = img['width']
    height= img['height']
    file_name  = img['file_name']
    gt_ajustado['images'].append({
      'id': img_id,
      'width': width,
      'height': height,
      'file_name': file_name
    })

  gt_ajustado['annotations'] = list()
  for ann in gt['annotations']:
    id = ann['id']
    image_id = ann['image_id']
    category_id= ann['category_id']
    bbox  = ann['bbox']
    area = ann['area']
    gt_ajustado['annotations'].append({
        'id': id,
        'image_id': image_id,
        'category_id': category_id,
        'bbox': bbox,
        'iscrowd':0,
        'area':area
    })

  gt_ajustado['categories'] = list()
  for category in gt['categories']:
    id = category['id']
    name = category['name']
    gt_ajustado['categories'].append({
        'id': id,
        'name': name
    })


  with open(caminho_arquivo_json, 'w') as arquivo:
      json.dump(gt_ajustado, arquivo, indent=4)
  return caminho_arquivo_json
  
def coco_metric(gt_json, predictions_json, thr_score= 0.0):
  predictions = thr_score_on_prediction(predictions_json, thr_score)
  gt_ajustado_path = ajust_ground_truth(gt_json)
  gt_coco = COCO(gt_ajustado_path)
  os.remove(gt_ajustado_path)
  pred_coco = gt_coco.loadRes(predictions)
  coco_eval = COCOeval(gt_coco, pred_coco, 'bbox')
  coco_eval.evaluate()
  coco_eval.accumulate()
  coco_eval.summarize()
  return coco_eval.stats[0]

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

def mIoU(coco_gt_path, coco_result_path, thr_score):
  with open(coco_gt_path, 'r') as f:
      gt = json.load(f)
  with open(coco_result_path, 'r') as f:
      predictions = json.load(f)

  predictions = thr_score_on_prediction(predictions, thr_score)
  gt = ajust_ground_truth(gt)

  # Carregar o arquivo COCO JSON do ground truth
  gt_coco = COCO(gt)
  ious = []
  for gt_ann in gt_coco.dataset['annotations']:
      image_id = gt_ann['image_id']
      gt_bbox = gt_ann['bbox']

      pred_anns = [ann for ann in predictions if ann['image_id'] == image_id] # Find all predictions for the current image
      # Compute iou for each prediction with the gt
      for pred_ann in pred_anns:
          pred_bbox = pred_ann['bbox']
          iou = bbox_iou(gt_bbox, pred_bbox)
          ious.append(iou)

  mean_iou = np.mean(ious)
  print(f'mIoU: {mean_iou}')
  return mean_iou

def verificar_todas_marcadas(list_ann):
  for ann in list_ann:
    if ann['Marked'] != True:
      return False
  return False
# Comparar cada bounding do gt com todas predições
# Salvar o melhor_iou e a predição q deu esse iou
# Depois de percorrer todaas predições
# Markar a melhor predição e o ann do groundtruth
# Assim que acabar de percorrer a lista de ann do groundth
# Verificar se existem predições sobrando
# Todas eu sobrarem são FP, pois marcam objetos que não existem, pois todos reais objetos já foram marcados junto a uma anotação do gt
# Se a lista de predições estiver vazia siginifica que todas anotações sobrando no groundtruth são FN, pois não há predições relativa a elas

# FN: Existem 2 sitações
# (1) Todas minhas predições ja foram marcadas (associadas a uma bounding box do gt)
# (2) Não achei nenhuma bounding box nas predições que tenha iou com o gt. Neste caso a lista de predições possui predições não marcadas.

# TP: Exite 1 situação
# (1) IoU >= thr e a mesma classe

# FP: Existem 2 situações
# (1) A predição não possuim IoU com o groundtruth. Todos elementos do groundtruth estão marcadas mas existem elementos não marcados na lista de predições.
# (2) O IoU é >= thr, porém a classe está errada.

def no_more_predictions(gt_anns):
  pred = []
  true = []

  for ann_left in gt_anns:
    true.append(ann_left["category_id"])
    pred.append(-1)
  return true, pred

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


# monta o y_true e y_pred para uma imagem a partir da lista de predições e gt dessa imagem
def generate_preds_true_vector_from_anns(gts, predictions_anns, iou_thr):
  y_pred = []
  y_true = []
  gt_anns = []
  pred_anns = predictions_anns.copy()
  gt_anns = gts.copy()

  i=0
  while i < len(gt_anns):
    ann_true = gt_anns[i]
    #Fn
    if len(pred_anns) == 0:
      t, p = no_more_predictions(gt_anns)
      y_pred.extend(p)
      y_true.extend(t)
      gt_anns = []
      break

    index_best_pred, best_iou = find_best_match(pred_anns, ann_true)
    if best_iou > iou_thr:
      pred_best = pred_anns[index_best_pred]
      class_pred = pred_best["category_id"]
      class_true = ann_true["category_id"]
      #melhor match tem a mesma classe logo TP
      if class_pred == class_true:
        #TP
        y_pred.append(class_pred)
        y_true.append(class_true)
        gt_anns.pop(i)
        pred_anns.pop(index_best_pred)
      # melhor match não tem a mesma classe logo foi por IoU, ou seja tem IoU> 0.5, ou seja achou o objeto porem com classe errada
      else:
        #FP
        y_pred.append(class_pred)
        y_true.append(class_true)
        gt_anns.pop(i)
        pred_anns.pop(index_best_pred)

    i+=1
  for pred in pred_anns:
    y_pred.append(pred["category_id"])
    y_true.append(-1)

  for gt in gt_anns:
    y_pred.append(-1)
    y_true.append(gt["category_id"])

  return y_true, y_pred

def get_all_bboxes_from_image_id(gt_coco, id):
   return [ann for ann in gt_coco['annotations'] if ann['image_id'] == id]

def thr_score_on_prediction(predictions_json, score):
  return [pred for pred in predictions_json if pred['score'] >= score]

def generate_true_and_pred_vector(coco_gt_path, coco_result_path, thr_score, iou_thr):
  with open(coco_gt_path, 'r') as f:
      gt_json = json.load(f)
  with open(coco_result_path, 'r') as f:
      predictions = json.load(f)
  predictions = thr_score_on_prediction(predictions, thr_score)
  gt_ajusted = ajust_ground_truth(gt_json)
  gt_coco = COCO(gt_ajusted)
  ids_das_imagens = [imagem["id"] for imagem in gt_json["images"]]
  y_true_full = []
  y_pred_full = []
  categories_anns = gt_json["categories"]
  classes = [c["id"] for c in categories_anns if c['id'] != 0]
  # classes = [c["id"] for c in categories_anns]

  for id in tqdm(ids_das_imagens):
    preds = [pred for pred in predictions if pred["image_id"] == id]
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

def calc_metrics(coco_gt_path, coco_result_path, thr_score=0.4, iou_thr=0.5):
  # TODO:
  # open jsons from coco_gt_path and coco_result_path in this function and pass the jsons to the functions below
  gt_json, predictions_json = open_jsons([coco_gt_path, coco_result_path])
  map = coco_metric(gt_json, predictions_json, thr_score= thr_score)
#   miou = mIoU(coco_gt_path, coco_result_path, thr_score)
#   y_true, y_pred, classes =generate_true_and_pred_vector(coco_gt_path, coco_result_path, thr_score, iou_thr)
#   f1_score = classification_metrics(y_pred, y_true,classes)
#   return (map + miou + f1_score)/3