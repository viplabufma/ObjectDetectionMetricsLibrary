from pycocotools.coco import COCO
from typing import Dict, List, Tuple, Optional, Union
from metrics import DetectionMetrics
import contextlib
import io

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
        return metrics_calculator.compute_metrics()
    
    def _process_all_images(self, metrics_calculator: DetectionMetrics) -> None:
        """Process all images through the metrics calculator"""
        img_ids = self.get_image_ids()
        for img_id in img_ids:
            gt_anns, pred_anns = self.get_annotations(img_id)
            metrics_calculator.process_image(gt_anns, pred_anns)

def print_metrics(metrics: dict, class_names: dict) -> None:
    """
    Print object detection metrics in an organized, readable format.
    
    Args:
        metrics: Metrics dictionary returned by DetectionMetrics.compute_metrics()
        class_names: Mapping of category_id to class name
    """
    # Formatting constants
    HEADER_WIDTH = 80
    METRIC_WIDTH = 20
    
    # Helper function to format metric values
    def format_metric(value, is_percentage=True):
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.4f}" if is_percentage else f"{value:.2f}"
        return str(value)
    
    # Helper function to print metric sections
    def print_section(title, data, is_class=False):
        print(f"\n{'=' * HEADER_WIDTH}")
        print(f"{title.upper():^{HEADER_WIDTH}}")
        print('=' * HEADER_WIDTH)
        
        if is_class:
            # Class metrics header
            print(f"{'Class':<25} | {'Precision':<{METRIC_WIDTH}} | {'Recall':<{METRIC_WIDTH}} | "
                  f"{'F1-Score':<{METRIC_WIDTH}} | {'Support':<{METRIC_WIDTH}}")
            print('-' * HEADER_WIDTH)
            
            # Print metrics for each class
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
            # Global metrics header
            global_data = data.get('global', {})
            print(f"{'METRIC':<30} | {'VALUE'}")
            print('-' * HEADER_WIDTH)
            
            # Print basic metrics
            for metric in ['precision', 'recall', 'f1', 'support']:
                if metric in global_data:
                    print(f"{metric.capitalize():<30} | {format_metric(global_data[metric])}")
            
            # Print mAP metrics if available
            map_metrics = {
                'mAP@0.5:0.95': global_data.get('mAP'),
                'mAP@0.5': global_data.get('mAP50'),
                'mAP@0.75': global_data.get('mAP75')
            }
            
            for metric, value in map_metrics.items():
                if value is not None:
                    print(f"{metric:<30} | {format_metric(value)}")
    
    # Print global metrics section
    print_section("Global Metrics", metrics)
    
    # Print per-class metrics section if available
    if any(cls_id != 'global' for cls_id in metrics.keys()):
        print_section("Per-Class Metrics", metrics, is_class=True)
    
    # Print final summary
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