'''
Author: Matheus Levy
Organization: Viplab - UFMA
GitHub: https://github.com/viplabufma/MatheusLevy_mestrado
'''

from pycocotools.coco import COCO
from typing import Dict, List, Tuple, Optional
from metrics import DetectionMetrics
import contextlib
import io
import json
import seaborn as sns;
import matplotlib.pyplot as plt
import json

class DetectionMetricsManager:
    """
    Manages the calculation of object detection metrics by comparing ground truth and prediction results.
    
    Attributes:
        gt_path (str): Path to ground truth JSON file (COCO format)
        result_path (str): Path to prediction results JSON file (COCO predictions format)
        gt_coco (COCO): COCO object containing ground truth data
        dt_coco (COCO): COCO object containing prediction data
        names (Dict[int, str]): Mapping of category_id to class name
    """
    
    def __init__(self, gt_path: str, result_path: str):
        self._initialize(gt_path, result_path)
        self.labels = []

    def _initialize(self, gt_path: str, result_path: str):
        """Initialize paths and empty data containers"""
        self.gt_path = gt_path
        self.result_path = result_path
        self.gt_coco: Optional[COCO] = None
        self.dt_coco: Optional[COCO] = None
        self.names: Dict[int, str] = {}

    def update_data(self, gt_path: str, result_path: str) -> None:
        """Update data sources and reload all data"""
        self._initialize(gt_path, result_path)
        self.load_data()

    def load_data(self) -> None:
        """Load and process ground truth and prediction JSON files"""
        # Suppress pycocotools output during loading
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            self._load_coco_data()
            self._extract_class_names()
        
    def _load_coco_data(self) -> None:
        """Load files using COCO API and prepare data"""
        self.gt_coco = COCO(self.gt_path)
        self.dt_coco = self.gt_coco.loadRes(self.result_path)
        
    def _extract_class_names(self) -> None:
        """Extract category_id to class name mapping"""
        self.names = {
            cat['id']: cat['name'] 
            for cat in self.gt_coco.dataset['categories']
        }
        
    def get_image_ids(self) -> List[int]:
        """Get list of image IDs present in ground truth"""
        return self.gt_coco.getImgIds()
    
    def get_annotations(self, img_id: int) -> Tuple[List[dict], List[dict]]:
        """
        Retrieve ground truth and prediction annotations for a specific image.
        
        Args:
            img_id: Image ID
            
        Returns:
            Tuple containing:
            - List of ground truth annotations
            - List of prediction annotations
        """
        gt_anns = self._load_ground_truth_annotations(img_id)
        pred_anns = self._load_prediction_annotations(img_id)
        return gt_anns, pred_anns
    
    def _load_ground_truth_annotations(self, img_id: int) -> List[dict]:
        """Load ground truth annotations for a specific image"""
        ann_ids = self.gt_coco.getAnnIds(imgIds=img_id)
        return self.gt_coco.loadAnns(ann_ids)
    
    def _load_prediction_annotations(self, img_id: int) -> List[dict]:
        """Load predictions for a specific image"""
        ann_ids = self.dt_coco.getAnnIds(imgIds=img_id)
        return self.dt_coco.loadAnns(ann_ids)
        
    def calculate_metrics(
        self, 
        iou_thr: float = 0.5, 
        conf_thr: float = 0.5,
        exclude_class: list = None) -> dict:
        """
        Calculate detection metrics for the entire dataset.
        
        Args:
            iou_thr: IoU threshold for true positive consideration
            conf_thr: Confidence threshold for prediction consideration
            exclude_class: List of class IDs to exclude from metrics
            
        Returns:
            Dictionary with calculated metrics
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
        self.labels = metrics_calculator.get_confusion_matrix_labels()
        metrics = metrics_calculator.compute_metrics()
        return {'confusion_matrix': metrics_calculator.matrix.tolist(), 'confusion_matrix_multiclass': metrics_calculator.multiclass_matrix.tolist(), **replace_keys_with_classes(metrics, self.labels)}
    
    def _process_all_images(self, metrics_calculator: DetectionMetrics) -> None:
        """Process all images through the metrics calculator"""
        img_ids = self.get_image_ids()
        for img_id in img_ids:
            gt_anns, pred_anns = self.get_annotations(img_id)
            metrics_calculator.process_image(gt_anns, pred_anns)

def save_confusion_matrix(
    matrix: list[list[float]], 
    class_names: list[str], 
    path: str = 'confusion_matrix.png',
    background_class=False
) -> None:
    """
    Salva uma matriz de confusão com nomes de classes personalizados.
    
    Args:
        matrix: Matriz de confusão (lista de listas)
        class_names: Lista com nomes das classes na ordem correta
        path: Caminho para salvar a imagem
    """
    if not background_class: class_names.append("background")
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        matrix,
        annot=True,
        annot_kws={"size": 12},
        fmt='g',
        cbar=False,
        cmap="viridis",
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def map_class_keys(metrics: dict, class_names: list) -> dict:
    """
    Mapeia chaves numéricas (IDs de classe) para nomes de classes em um dicionário de métricas.
    
    Args:
        metrics: Dicionário de métricas com chaves numéricas para classes
        class_names: Lista de nomes de classes na ordem dos IDs
        
    Returns:
        Dicionário com chaves numéricas substituídas por nomes de classes
    """
    mapped_metrics = {}
    class_id_to_name = {i: name for i, name in enumerate(class_names)}
    
    for key, value in metrics.items():
        # Substitui chaves numéricas por nomes de classes
        if isinstance(key, int) and key in class_id_to_name:
            new_key = class_id_to_name[key]
            mapped_metrics[new_key] = value
        # Mantém chaves não numéricas (global, confusion_matrix, etc)
        else:
            mapped_metrics[key] = value
            
    return mapped_metrics

def export_metrics(metrics: dict, class_names: list, path: str = '.', format: str = 'json'):
    """
    Exporta métricas com nomes de classes para um arquivo.
    
    Args:
        metrics: Dicionário de métricas original
        class_names: Lista de nomes de classes na ordem dos IDs
        path: Diretório de saída
        format: Formato do arquivo (apenas 'json' suportado)
    """
    # Mapeia IDs numéricos para nomes de classes
    mapped_metrics = map_class_keys(metrics, class_names)
    
    with open(f"{path}/metrics.{format}", 'w') as f:
        if format == 'json':
            json.dump(mapped_metrics, f, indent=4)
        else:
            raise ValueError("Unsupported format. Use 'json'.")

def replace_keys_with_classes(metrics_dict, class_list):
    """
    Replaces numeric keys in a metrics dictionary with corresponding class names.
    
    This function processes dictionary keys that represent class indices (as integers 
    or numeric strings) and replaces them with the corresponding class name from 
    the provided class list. Non-numeric keys are preserved unchanged.
    
    Args:
        metrics_dict (dict): Input dictionary containing metric data. Keys may be:
            - Integers representing class indices
            - Numeric strings representing class indices
            - Non-numeric keys (e.g., 'global') to be preserved
        class_list (list): List of class names where index corresponds to class ID
    
    Returns:
        dict: New dictionary with numeric keys replaced by class names where applicable
        
    Example:
        >>> metrics_dict = {
                0: {'precision': 0.666,...},
                'global': {'precision': 0.666,...}
            }
        >>> class_list = ['person', 'background']
        >>> replace_keys_with_classes(metrics_dict, class_list)
        {
            'person': {'precision': 0.666,...},
            'global': {'precision': 0.666,...}
        }
    """
    new_dict = {}
    for key, value in metrics_dict.items():
        # Attempt to process numeric keys (both integers and string representations)
        try:
            # Convert key to integer (works for both int and numeric strings)
            index = int(key)
            
            # Check if index is within valid range of class_list
            if 0 <= index < len(class_list):
                # Replace with corresponding class name
                new_dict[class_list[index]] = value
            else:
                # Keep original key if index is out of range
                new_dict[key] = value
                
        except (ValueError, TypeError):
            # Keep non-numeric keys unchanged (e.g., 'global')
            new_dict[key] = value
            
    return new_dict