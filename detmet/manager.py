"""
Author: Matheus Levy
Organization: Viplab - UFMA
GitHub: https://github.com/viplabufma/MatheusLevy_mestrado
"""
from pycocotools.coco import COCO
from typing import Dict, List, Tuple, Optional
from .results import MetricsResult
from .metrics import DetectionMetrics
from .utils import check_normalized, map_class_keys_recursive
import contextlib
import io

class DetectionMetricsManager:
    """
    Manages the calculation of object detection metrics by comparing ground truth and prediction results.
    
    This class handles loading COCO-format datasets, processing images, and computing various detection metrics
    including precision, recall, F1-score, confusion matrices, and COCO-style mAP.
    
    Attributes
    ----------
    groundtruth_json_path : str
        Path to ground truth JSON file (COCO format)
    prediction_json_path : str
        Path to prediction results JSON file (COCO predictions format)
    gt_coco : COCO
        COCO object containing ground truth data
    dt_coco : COCO
        COCO object containing prediction data
    names : Dict[int, str]
        Mapping of category_id to class name
    labels : List[str]
        Class labels including background

    Examples
    --------
    >>> manager = DetectionMetricsManager('path/to/gt.json', 'path/to/predictions.json')
    >>> metrics = manager.calculate_metrics()
    >>> metrics.plot_confusion_matrix('confusion.png')
    """
    
    def __init__(self, groundtruth_json_path: str, prediction_json_path: str):
        """
        Initialize DetectionMetricsManager with ground truth and prediction paths.
        
        Parameters
        ----------
        groundtruth_json_path : str
            Path to COCO-format ground truth JSON file
        prediction_json_path : str
            Path to COCO-format prediction results JSON file
        """
        self._initialize(groundtruth_json_path, prediction_json_path)
        self.labels = []

    def _initialize(self, groundtruth_json_path: str, prediction_json_path: str) -> None:
        """Initialize paths and empty data containers"""
        self.gt_path = groundtruth_json_path
        self.pred_path = prediction_json_path
        self.gt_coco: Optional[COCO] = None
        self.dt_coco: Optional[COCO] = None
        self.names: Dict[int, str] = {}
        self._load_data()

    def update_data(self, groundtruth_json_path: str, prediction_json_path: str) -> None:
        """
        Update data sources and reload all data.
        
        Parameters
        ----------
        gt_path : str
            New path to ground truth JSON file
        result_path : str
            New path to prediction results JSON file
        """
        self._initialize(groundtruth_json_path, prediction_json_path)
        self._load_data()

    def _load_data(self) -> None:
        """Load and process ground truth and prediction JSON files"""
        # Suppress pycocotools output during loading
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            self._load_coco_data()
            self._extract_class_names()
        
    def _load_coco_data(self) -> None:
        """Load files using COCO API and prepare data"""
        try:
            self.gt_coco = COCO(self.gt_path)
            self.dt_coco = self.gt_coco.loadRes(self.pred_path)
        except Exception as e:
            raise ValueError("Invalid JSON file") from e
        
    def _extract_class_names(self) -> None:
        """Extract category_id to class name mapping"""
        self.names = {
            cat['id']: cat['name'] 
            for cat in self.gt_coco.dataset['categories']
        }
        
    def get_image_ids(self) -> List[int]:
        """
        Get list of image IDs present in ground truth.
        
        Returns
        -------
        List[int]
            List of image IDs in the dataset
        """
        return self.gt_coco.getImgIds()
    
    def get_annotations(self, img_id: int) -> Tuple[List[dict], List[dict]]:
        """
        Retrieve ground truth and prediction annotations for a specific image.
        
        Parameters
        ----------
        img_id : int
            Image ID to retrieve annotations for
            
        Returns
        -------
        Tuple[List[dict], List[dict]]
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
    
    def calculate_metrics(self, iou_thr: float = 0.5, conf_thr: float = 0.5, 
                          exclude_classes: Optional[list] = None) -> MetricsResult:
        """
        Calculate detection metrics for the loaded dataset.
        
        Parameters
        ----------
        iou_thr : float, optional
            IoU threshold for true positive matching (0.0-1.0), by default 0.5
        conf_thr : float, optional
            Confidence threshold for predictions (0.0-1.0), by default 0.5
        exclude_classes : Optional[list], optional
            List of class IDs to exclude from evaluation, by default None
            
        Returns
        -------
        MetricsResult
            Object containing computed metrics and visualization methods
            
        Raises
        ------
        ValueError
            If iou_thr or conf_thr are not in [0.0, 1.0]
            
        Examples
        --------
        >>> manager = DetectionMetricsManager('gt.json', 'preds.json')
        >>> result = manager.calculate_metrics(iou_thr=0.5, conf_thr=0.5)
        >>> print(result.metrics['global']['precision'])
        """
        check_normalized(iou_thr)
        check_normalized(conf_thr)
        metrics_calculator = DetectionMetrics(
            names=self.names,
            iou_thr=iou_thr,
            conf_thr=conf_thr,
            gt_coco=self.gt_coco,
            predictions_coco=self.dt_coco,
            exclude_classes=exclude_classes,
            store_pr_data=True,
            store_pr_curves=True
        )
        self._process_all_images(metrics_calculator)
        labels = metrics_calculator.get_confusion_matrix_labels()
        metrics = metrics_calculator.compute_metrics()
        metrics = map_class_keys_recursive(metrics, labels)
        return MetricsResult(
            metrics={
                'confusion_matrix': metrics_calculator.matrix.tolist(),
                'confusion_matrix_multiclass': metrics_calculator.multiclass_matrix.tolist(),
                **metrics
            },
            labels=labels
        )
    
    def _process_all_images(self, metrics_calculator: DetectionMetrics) -> None:
        """Process all images through the metrics calculator"""
        img_ids = self.get_image_ids()
        for img_id in img_ids:
            gt_anns, pred_anns = self.get_annotations(img_id)
            metrics_calculator.process_image(gt_anns, pred_anns)
    
def compute_metrics(groundtruth_json_path: str, prediction_json_path: str, 
                   iou_thr: float = 0.5, conf_thr: float = 0.0, 
                   exclude_classes: list = None) -> None:
    """
    Compute and export detection metrics with visualizations.
    
    This function provides a simplified interface for:
    1. Computing detection metrics
    2. Saving confusion matrix plot
    3. Exporting metrics to JSON
    4. Plotting PR curves
    
    Parameters
    ----------
    groundtruth_json_path : str
        Path to COCO-format ground truth JSON file
    prediction_json_path : str
        Path to COCO-format prediction results JSON file
    iou_thr : float, optional
        IoU threshold for true positive matching (0.0-1.0), by default 0.5
    conf_thr : float, optional
        Confidence threshold for predictions (0.0-1.0), by default 0.0
    exclude_classes : list, optional
        List of class IDs to exclude from evaluation, by default None
    
    Raises
    ------
    ValueError
        If iou_thr or conf_thr are not in [0.0, 1.0]
    
    Examples
    --------
    >>> compute_metrics(
    ...     'path/to/gt.json',
    ...     'path/to/predictions.json',
    ...     iou_thr=0.5,
    ...     conf_thr=0.5
    ... )
    # Generates:
    #   - confusion_matrix.png
    #   - metrics.json
    #   - pr_curves.png
    """
    check_normalized(iou_thr)
    check_normalized(conf_thr)
    manager = DetectionMetricsManager(groundtruth_json_path=groundtruth_json_path,prediction_json_path=prediction_json_path)
    metrics = manager.calculate_metrics(conf_thr=conf_thr, iou_thr=iou_thr, exclude_classes=exclude_classes)
    metrics.plot_confusion_matrix('confusion_matrix.png')
    metrics.export()
    metrics.plot_pr_curves()