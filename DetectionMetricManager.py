import json
from pycocotools.coco import COCO
from typing import Dict, List, Tuple, Optional
from metrics import DetectionMetrics
import contextlib
import io

class DetectionMetricsManager:
    """
    Gerencia o cálculo de métricas de detecção de objetos comparando ground truth e resultados.
    
    Attributes:
        gt_path (str): Caminho para o arquivo JSON de ground truth (formato COCO)
        result_path (str): Caminho para o arquivo JSON de resultados (formato COCO predictions)
        gt_coco (COCO): Objeto COCO com dados de ground truth
        dt_coco (COCO): Objeto COCO com dados de predições
        names (Dict[int, str]): Mapeamento de category_id para nome da classe
    """
    
    def __init__(self, gt_path: str, result_path: str):
        self._initialize(gt_path, result_path)
    
    def _initialize(self, gt_path: str, result_path: str):
        self.gt_path = gt_path
        self.result_path = result_path
        self.gt_coco: Optional[COCO] = None
        self.dt_coco: Optional[COCO] = None
        self.names: Dict[int, str] = {}

    def update_data(self, gt_path: str, result_path: str) -> None:
        self._initialize(gt_path, result_path)
        self.load_data()

    def load_data(self) -> None:
        """Carrega e processa os arquivos JSON de ground truth e resultados."""
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            self._load_coco_data()
            self._extract_class_names()
        
    def _load_coco_data(self) -> None:
        """Carrega os arquivos usando a API COCO e prepara os dados."""
        self.gt_coco = COCO(self.gt_path)
        self.dt_coco = self.gt_coco.loadRes(self.result_path)
        
    def _extract_class_names(self) -> None:
        """Extrai o mapeamento de category_id para nome de classe."""
        self.names = {
            cat['id']: cat['name'] 
            for cat in self.gt_coco.dataset['categories']
        }
        
    def get_image_ids(self) -> List[int]:
        """Obtém a lista de IDs de imagens presentes no ground truth."""
        return self.gt_coco.getImgIds()
    
    def get_annotations(self, img_id: int) -> Tuple[List[dict], List[dict]]:
        """
        Obtém anotações de ground truth e predições para uma imagem específica.
        
        Args:
            img_id: ID da imagem
            
        Returns:
            Tuple com:
            - Lista de anotações de ground truth
            - Lista de predições
        """
        gt_anns = self._load_ground_truth_annotations(img_id)
        pred_anns = self._load_prediction_annotations(img_id)
        return gt_anns, pred_anns
    
    def _load_ground_truth_annotations(self, img_id: int) -> List[dict]:
        """Carrega anotações de ground truth para uma imagem específica."""
        ann_ids = self.gt_coco.getAnnIds(imgIds=img_id)
        return self.gt_coco.loadAnns(ann_ids)
    
    def _load_prediction_annotations(self, img_id: int) -> List[dict]:
        """Carrega predições para uma imagem específica."""
        ann_ids = self.dt_coco.getAnnIds(imgIds=img_id)
        return self.dt_coco.loadAnns(ann_ids)
        
    def calculate_metrics(
        self, 
        iou_thr: float = 0.5, 
        conf_thr: float = 0.5,
        exclude_class= None
    ) -> dict:
        """
        Calcula métricas de detecção para todo o dataset.
        
        Args:
            iou_thr: Limiar de IoU para consideração de verdadeiro positivo
            conf_thr: Limiar de confiança para consideração de predições
            
        Returns:
            Dicionário com métricas calculadas
        """
        metrics_calculator = DetectionMetrics(
            names=self.names,
            iou_thr=iou_thr,
            conf_thr=conf_thr,
            gt_coco=self.gt_coco,
            predictions_coco=self.dt_coco,
            exclude_classes=exclude_class
        )
        
        self._process_all_images(metrics_calculator)
        return metrics_calculator.compute_metrics()
    
    def _process_all_images(self, metrics_calculator: DetectionMetrics) -> None:
        """Processa todas as imagens através da calculadora de métricas."""
        img_ids = self.get_image_ids()
        for img_id in img_ids:
            gt_anns, pred_anns = self.get_annotations(img_id)
            metrics_calculator.process_image(gt_anns, pred_anns)

def print_metrics(metrics: dict, class_names: dict, title: str = "Detection Metrics") -> None:
    """
    Imprime métricas de detecção de objetos de forma organizada e legível.
    
    Args:
        metrics (dict): Dicionário de métricas retornado por DetectionMetrics.compute_metrics()
        class_names (Dict[int, str]): Mapeamento de category_id para nome da classe
        title (str): Título para o cabeçalho da impressão
    """
    # Constantes de formatação
    HEADER_WIDTH = 80
    METRIC_WIDTH = 20
    VALUE_WIDTH = 10
    
    # Funções auxiliares
    def format_metric(value, is_percentage=True):
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.4f}" if is_percentage else f"{value:.2f}"
        return str(value)
    
    def print_section(title, data, is_class=False):
        print(f"\n{'=' * HEADER_WIDTH}")
        print(f"{title.upper():^{HEADER_WIDTH}}")
        print('=' * HEADER_WIDTH)
        
        if is_class:
            # Cabeçalho para métricas de classe
            print(f"{'Class':<25} | {'Precision':<{METRIC_WIDTH}} | {'Recall':<{METRIC_WIDTH}} | "
                  f"{'F1-Score':<{METRIC_WIDTH}} | {'Support':<{METRIC_WIDTH}}")
            print('-' * HEADER_WIDTH)
            
            for cls_id, cls_data in data.items():
                if cls_id == 'global':
                    continue
                    
                cls_name = class_names.get(cls_id, f'Class_{cls_id}')
                print(f"{cls_name:<25} | "
                      f"{format_metric(cls_data.get('precision')):<{METRIC_WIDTH}} | "
                      f"{format_metric(cls_data.get('recall')):<{METRIC_WIDTH}} | "
                      f"{format_metric(cls_data.get('f1')):<{METRIC_WIDTH}} | "
                      f"{format_metric(cls_data.get('support'), False):<{METRIC_WIDTH}}")
        else:
            # Métricas globais
            global_data = data.get('global', {})
            print(f"{'METRIC':<30} | {'VALUE'}")
            print('-' * HEADER_WIDTH)
            
            # Métricas básicas
            for metric in ['precision', 'recall', 'f1', 'support']:
                if metric in global_data:
                    print(f"{metric.capitalize():<30} | {format_metric(global_data[metric])}")
            
            # Métricas mAP
            map_metrics = {
                'mAP@0.5:0.95': global_data.get('mAP'),
                'mAP@0.5': global_data.get('mAP50'),
                'mAP@0.75': global_data.get('mAP75')
            }
            
            for metric, value in map_metrics.items():
                if value is not None:
                    print(f"{metric:<30} | {format_metric(value)}")
    
    # Imprimir métricas globais
    print_section("Global Metrics", metrics)
    
    # Imprimir métricas por classe
    if any(cls_id != 'global' for cls_id in metrics.keys()):
        print_section("Per-Class Metrics", metrics, is_class=True)
    
    # Resumo final
    global_data = metrics.get('global', {})
    if 'mAP' in global_data:
        print(f"\n{' SUMMARY ':=^{HEADER_WIDTH}}")
        print(f"→ mAP@0.5:0.95: {format_metric(global_data['mAP'])}")
        print(f"→ mAP@0.5:      {format_metric(global_data.get('mAP50'))}")
        print(f"→ mAP@0.75:     {format_metric(global_data.get('mAP75'))}")
        print(f"→ Precision:    {format_metric(global_data.get('precision'))}")
        print(f"→ Recall:       {format_metric(global_data.get('recall'))}")
        print(f"→ F1-Score:     {format_metric(global_data.get('f1'))}")
        print(f"→ Support:      {format_metric(global_data.get('support'), False)}")
        print('=' * HEADER_WIDTH)

if __name__ == "__main__":
    manager = DetectionMetricsManager(
        gt_path="/home/thebig/Documentos/MatheusLevy_mestrado/tests/jsons/_annotations.coco.json",
        result_path="/home/thebig/Documentos/MatheusLevy_mestrado/tests/jsons/tood_predicts_bbox.bbox.json"
    )
    manager.load_data()
    # manager.update_data(
    #     gt_path="/home/thebig/Documentos/MatheusLevy_mestrado/tests/jsons/_annotations.coco.json",
    #     result_path="/home/thebig/Documentos/MatheusLevy_mestrado/tests/jsons/tood_predicts_bbox.bbox.json"
    # )
    metrics = manager.calculate_metrics(exclude_class=[0])
    custom_metrics = manager.calculate_metrics(iou_thr=0.5, conf_thr=0.3, exclude_class=[0])
    print_metrics(custom_metrics, manager.names, title="Custom Metrics (IoU=0.5, Conf=0.3)")
