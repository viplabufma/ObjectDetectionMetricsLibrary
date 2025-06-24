"""
Author: Matheus Levy
Organization: Viplab - UFMA
GitHub: https://github.com/viplabufma/MatheusLevy_mestrado
"""

import os
import numpy as np
from pycocotools.coco import COCO
from typing import Dict, List, Tuple, Optional, Union, Any
from .metrics import DetectionMetrics
import contextlib
import io
import json
import seaborn as sns
import matplotlib.pyplot as plt


class DetectionMetricsManager:
    """
    Manages the calculation of object detection metrics by comparing ground truth and prediction results.
    
    Attributes:
        gt_path (str): Path to ground truth JSON file (COCO format)
        result_path (str): Path to prediction results JSON file (COCO predictions format)
        gt_coco (COCO): COCO object containing ground truth data
        dt_coco (COCO): COCO object containing prediction data
        names (Dict[int, str]): Mapping of category_id to class name
        labels (List[str]): Class labels including background
    """
    
    def __init__(self, gt_path: str, result_path: str):
        self._initialize(gt_path, result_path)
        self.labels = []

    def _initialize(self, gt_path: str, result_path: str) -> None:
        """Initialize paths and empty data containers"""
        self.gt_path = gt_path
        self.result_path = result_path
        self.gt_coco: Optional[COCO] = None
        self.dt_coco: Optional[COCO] = None
        self.names: Dict[int, str] = {}
        self._load_data()

    def update_data(self, gt_path: str, result_path: str) -> None:
        """Update data sources and reload all data"""
        self._initialize(gt_path, result_path)
        self._load_data()

    def _load_data(self) -> None:
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
        exclude_class: Optional[list] = None
    ) -> dict:
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
            exclude_classes=exclude_class,
            store_pr_data=True,
            store_pr_curves=True
        )
        self._process_all_images(metrics_calculator)
        self.labels = metrics_calculator.get_confusion_matrix_labels()
        metrics = metrics_calculator.compute_metrics()
        metrics = map_class_keys_recursive(metrics, self.labels)
        return {
            'confusion_matrix': metrics_calculator.matrix.tolist(),
            'confusion_matrix_multiclass': metrics_calculator.multiclass_matrix.tolist(),
            **metrics
        }
    
    def _process_all_images(self, metrics_calculator: DetectionMetrics) -> None:
        """Process all images through the metrics calculator"""
        img_ids = self.get_image_ids()
        for img_id in img_ids:
            gt_anns, pred_anns = self.get_annotations(img_id)
            metrics_calculator.process_image(gt_anns, pred_anns)


def save_confusion_matrix(
    matrix: List[List[float]], 
    class_names: List[str], 
    path: str = 'confusion_matrix.png',
    background_class: bool = False
) -> None:
    """
    Save a confusion matrix with custom class names.
    
    Args:
        matrix: Confusion matrix (list of lists)
        class_names: List of class names in correct order
        path: Path to save the image
        background_class: Whether to include background class
    """
    if background_class:
        class_names.append("background")
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


def map_class_keys_recursive(
    obj: Union[dict, list, Any], 
    class_list: List[str]
) -> Union[dict, list, Any]:
    """
    Recursively maps numeric keys to class names throughout all dictionary levels.
    
    Args:
        obj: Dictionary or object to process
        class_list: List of class names
    
    Returns:
        Object with numeric keys replaced by class names
    """
    if class_list and class_list[-1] == 'background':
        class_list = class_list[:-1]
    
    if isinstance(obj, dict):
        # Process dictionaries
        new_dict = {}
        
        # Separate numeric and non-numeric keys
        numeric_keys = []
        non_numeric_items = {}
        
        for key, value in obj.items():
            try:
                # Try to convert key to int
                num_key = int(key)
                numeric_keys.append((key, num_key, value))
            except (ValueError, TypeError):
                # Non-numeric key - process value recursively
                non_numeric_items[key] = map_class_keys_recursive(value, class_list)
        
        # Sort numeric keys
        numeric_keys.sort(key=lambda x: x[1])
        
        # Map numeric keys to class names
        for idx, (orig_key, num_key, value) in enumerate(numeric_keys):
            if idx < len(class_list):
                new_key = class_list[idx]
            else:
                new_key = orig_key  # Keep original if no matching name
            
            new_dict[new_key] = map_class_keys_recursive(value, class_list)
        
        # Add non-numeric items
        new_dict.update(non_numeric_items)
        return new_dict
    
    elif isinstance(obj, list):
        # Process lists
        return [map_class_keys_recursive(item, class_list) for item in obj]
    
    else:
        # Keep other types unchanged
        return obj


def export_metrics(
    metrics: dict, 
    path: str = '.', 
    format: str = 'json'
) -> None:
    """
    Export metrics dictionary to file in specified format.
    
    Args:
        metrics: Metrics dictionary to export
        path: Output directory path
        format: File format ('json' supported)
    """
    # Convert NumPy objects to native types
    metrics_converted = convert_numpy(metrics)
    # Export to JSON
    with open(f"{path}/metrics.{format}", 'w') as f:
        if format == 'json':
            json.dump(metrics_converted, f, indent=4)
        else:
            raise ValueError("Unsupported format. Use 'json'.")


def replace_keys_with_classes(
    metrics_dict: dict, 
    class_list: List[str]
) -> dict:
    """
    Replaces numeric keys in a metrics dictionary with corresponding class names.
    Removes last class if it's 'background' before processing.
    
    Args:
        metrics_dict: Input dictionary with numeric/non-numeric keys
        class_list: List of class names (last element removed if 'background')
    
    Returns:
        New dictionary with numeric keys replaced by class names
        
    Example:
        >>> metrics_dict = {'a': 100, '0': 200, '1': 300}
        >>> class_list = ['person', 'background']
        >>> replace_keys_with_classes(metrics_dict, class_list)
        {'a': 100, 'person': 200}
    """
    # Remove last element if it's 'background'
    if class_list and class_list[-1] == 'background':
        class_list = class_list[:-1]
    
    # Separate numeric and non-numeric keys
    numeric_keys = []
    non_numeric_items = {}
    
    for key, value in metrics_dict.items():
        try:
            # Convert to int and add to numeric keys
            numeric_keys.append((key, int(key)))
        except (ValueError, TypeError):
            # Non-numeric key - preserve as-is
            non_numeric_items[key] = value
    
    # Sort numeric keys by their numeric value
    numeric_keys.sort(key=lambda x: x[1])
    sorted_numeric_keys = [orig_key for orig_key, _ in numeric_keys]
    
    # Build new dictionary
    new_dict = {}
    # Add non-numeric items first
    new_dict.update(non_numeric_items)
    # Add class-mapped items
    for idx, orig_key in enumerate(sorted_numeric_keys):
        if idx < len(class_list):
            new_dict[class_list[idx]] = metrics_dict[orig_key]
        else:
            new_dict[orig_key] = metrics_dict[orig_key]
            
    return new_dict


def convert_numpy(
    obj: Any, 
    convert_keys_to_str: bool = False
) -> Any:
    """
    Convert NumPy objects to native Python types recursively.
    
    Args:
        obj: Object to convert (any type)
        convert_keys_to_str: Convert non-string keys to strings (useful for JSON)
    
    Returns:
        Object converted to native Python types
    """
    # Handle arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # Handle NumPy scalars
    elif isinstance(obj, np.generic):
        if isinstance(obj, (np.integer, np.unsignedinteger)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.character):
            return str(obj)
        return obj.item()  # Fallback for other types
    
    # Handle dictionaries (keys and values)
    elif isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            k = convert_numpy(k, convert_keys_to_str)
            v = convert_numpy(v, convert_keys_to_str)
            if convert_keys_to_str and not isinstance(k, str):
                k = str(k)
            new_dict[k] = v
        return new_dict
    
    # Handle lists and tuples
    elif isinstance(obj, list):
        return [convert_numpy(item, convert_keys_to_str) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy(item, convert_keys_to_str) for item in obj)
    
    # Other types don't need conversion
    return obj


def plot_pr_curves(
    pr_curves: Dict[Union[int, str], Dict[str, np.ndarray]], 
    output_path: Optional[str] = None,
    show: bool = True,
    dpi: int = 100
) -> Optional[plt.Figure]:
    """
    Plot Precision-Recall curves from precomputed PR curves data.
    
    Args:
        pr_curves: Dictionary containing PR curves data
        output_path: Path to save the plot image
        show: Whether to display the plot
        dpi: Image resolution for saved figure
        
    Returns:
        plt.Figure if show=False, else None
    """
    if not pr_curves:
        raise ValueError("PR curves dictionary is empty")
    
    plt.figure(figsize=(12, 8))
    plt.title('Precision-Recall Curves')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.05)
    
    # Plot global curve
    global_data = pr_curves.get('global')
    if global_data:
        plt.plot(
            global_data['recall'], 
            global_data['precision'],
            'k-', 
            linewidth=3,
            label=f'Global (AP={global_data["ap"]:.3f})'
        )
    
    # Plot class curves
    for class_id, curve_data in pr_curves.items():
        if class_id == 'global':
            continue
            
        plt.plot(
            curve_data['recall'], 
            curve_data['precision'],
            label=f'{class_id} (AP={curve_data["ap"]:.3f})'
        )
    
    plt.legend(loc='lower left', fontsize=10)
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    
    if show:
        plt.show()
        return None
    else:
        return plt.gcf()