from utils import calc_metrics

coco_gt_path = r'C:\Users\levyb\Documents\MatheusLevy_mestrado\tests\jsons\_annotations.coco.json'
coco_result_path = r'C:\Users\levyb\Documents\MatheusLevy_mestrado\tests\jsons\tood_predicts_bbox.bbox.json'
calc_metrics(coco_gt_path, coco_result_path, thr_score= 0.4, iou_thr=0.5)