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
    Save a confusion matrix visualization with custom class names.
    
    Parameters
    ----------
    matrix : List[List[float]]
        Confusion matrix as a 2D list of numerical values
    class_names : List[str]
        List of class names in the order corresponding to matrix indices
    path : str, optional
        Output file path for the confusion matrix image, 
        by default 'confusion_matrix.png'
    background_class : bool, optional
        Whether to include background class in the visualization, 
        by default False
        
    Notes
    -----
    - The confusion matrix should be a square matrix
    - Class names should be in the same order as the matrix rows/columns
    - If background_class=True, appends "background" to class_names
    - Uses seaborn's heatmap for visualization with viridis colormap
    - Automatically rotates x-axis labels for better readability
    - Saves high-quality image with tight layout
    
    Examples
    --------
    >>> matrix = [[50, 2, 3], [1, 45, 4], [2, 3, 55]]
    >>> classes = ['cat', 'dog', 'bird']
    >>> save_confusion_matrix(matrix, classes, 'confusion.png')
    
    >>> # With background class
    >>> save_confusion_matrix(matrix, classes, 'confusion_bg.png', background_class=True)
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
    
    Parameters
    ----------
    pr_curves : Dict[Union[int, str], Dict[str, np.ndarray]]
        Dictionary containing PR curves data with structure:
        - 'global': Global PR curve data
            - 'recall': Recall values
            - 'precision': Precision values
            - 'ap': Average precision
        - Class IDs: Per-class PR curve data with same structure
    output_path : Optional[str], optional
        Path to save the plot image. If None, plot is not saved, 
        by default None
    show : bool, optional
        Whether to display the plot interactively, by default True
    dpi : int, optional
        Image resolution in dots per inch for saved figures, 
        by default 100
        
    Returns
    -------
    Optional[plt.Figure]
        Matplotlib figure object if show=False, else None
        
    Raises
    ------
    ValueError
        If pr_curves dictionary is empty
        
    Notes
    -----
    - Global curve is plotted in black with thicker line
    - Each class curve is plotted with its class ID in the legend
    - Legend includes Average Precision (AP) values
    - Plot includes grid and proper axis limits
    
    Examples
    --------
    >>> # Basic usage: display and save
    >>> plot_pr_curves(pr_data, 'pr_curves.png')
    
    >>> # Display without saving
    >>> plot_pr_curves(pr_data, show=True, output_path=None)
    
    >>> # Get figure without displaying
    >>> fig = plot_pr_curves(pr_data, show=False)
    >>> fig.savefig('custom.png', dpi=300)
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
    
    Parameters
    ----------
    metrics : dict
        Metrics dictionary to export
    path : str, optional
        Output directory path, by default current directory ('.')
    format : str, optional
        File format ('json' supported), by default 'json'
        
    Raises
    ------
    ValueError
        If unsupported format is requested
    OSError
        If there are issues writing to the specified path
        
    Notes
    -----
    - Converts NumPy objects to native Python types before serialization
    - Creates directory structure if needed
    - Output file will be named "metrics.{format}"
    - JSON output is pretty-printed with 4-space indentation
    
    Examples
    --------
    >>> # Export to JSON in current directory
    >>> export_metrics(metrics)
    
    >>> # Export to specific directory
    >>> export_metrics(metrics, path='output/metrics')
    
    >>> # Custom file format (not implemented)
    >>> export_metrics(metrics, format='yaml')
    Traceback (most recent call last):
    ...
    ValueError: Unsupported format. Use 'json'.
    """
    # Convert NumPy objects to native types
    metrics_converted = convert_numpy(metrics)
    
    # Create output directory if needed
    os.makedirs(path, exist_ok=True)
    
    # Export to JSON
    with open(os.path.join(path, f"metrics.{format}"), 'w') as f:
        if format == 'json':
            json.dump(metrics_converted, f, indent=4)
        else:
            raise ValueError("Unsupported format. Use 'json'.")