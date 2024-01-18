# Object Detection Metrics
## Matheus Levy 
## Organization: UFMA

## How to use:
```
from utils import open_jsons, ajust_ground_truth
from object_detection_metrics import mIoU, coco_metric, precision_recall_f1_score, false_negatives, acuracy, balanced_accuracy

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
```

## util Functions
* open_jsons(array_of _paths): read jsons and return the jsons
* ajust_ground_truth(gt_json): make a simplified json with only the infos needed.
 
 ## Parameters:
 * thr_score: threshold of score. All detections bellow that threshold will be discarted.
 * iou_thr: threshold of iou. The threshold to two bounding boxes be considerar as match (or refering to the same object)
 * skip_classes: a array of the classes wanted to be skiped in the metric.
 * average: how to compute some metrics. It can be 'average', 'macro' and 'micro'.

 
|         Metrics        |           Function          |
|:----------------------:|:---------------------------:|
|  Intersect Over Union  |            mIoU()           |
| Mean Average Precision |        coco_metric()        |
| Precision              | precision_recall_f1_score() |
| Recall                 | precision_recall_f1_score() |
| F1 Score               | precision_recall_f1_score() |
| False Negative Rate    |       false_negative()      |
| Accuracy               |          accuracy()         |
| Balanced Accuracy      |     balanced_accuracy()     |