import json
import os
from typing import Dict, List, Optional, Union
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from detmet.utils import convert_numpy

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
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    
    if show:
        plt.show()
        return None
    else:
        return plt.gcf()
    
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