# Detecção de Ovos de Parasitas em Imagens de Microscópio
## Matheus Levy

1. Avaliação de Modelos de Deep Learning para detecção (Feito)
  1.1. Faster RCNN, Cascade RCNN, YOLO,. etc (Feito)
  1.2. Métrica: mAP (api da coco)  (Feito)
2. Otimização de Detectores (Feito)
  2.1. Resultado: Cascade RCNN (HRNet), RetinaNet (Resnetx101 32), TOOD (ReseneXt 101-64 dcn)
3. Ensemble de Detectores (Feito)
   3.1. Resultado do Ensemble: 0,911
5. Novas métricas de avaliação (Fazendo)
6. Novo Ensemble (Fazendo)
7. Testar outras técnicas de Ensemble (A fazer)
   7.1. NMS, Soft-NMS
   
Estrutura das Pastas:

-> MMdetection Models and Checkpoints/
    -> 640_faster_rcnn_x101_32x8d_fpn_mstrain/
        -> faster_rcnn_x101_32x8d_fpn_mstrain_3x_coco_20210604_182954-002e082a.pth (checkpoint)
        -> predicts_bbox.bbox.json (predições)
        -> 640_640_faster_rcnn_x101_32x8d_fpn_mstrain.py (config/model)
    -> cascade-rcnn_x101-32x4d_fpn_20e_coco/
        -> cascade_rcnn_x101_32x4d_fpn_20e_coco_20200906_134608-9ae0a720.pth (checkpoint)
        -> predicts_bbox.bbox.json (predições)
        -> cascade-rcnn_x101-32x4d_fpn_20e_coco_custom.py (config/model)
    ...
    ...
    ...

