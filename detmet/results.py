from typing import List, Optional
from matplotlib import pyplot as plt
from .visualization import save_confusion_matrix, plot_pr_curves, export_metrics

class MetricsResult:
    """
    Encapsulates computed detection metrics with methods for visualization and export.
    
    This class provides a convenient interface for visualizing and exporting object detection
    evaluation metrics including confusion matrices, precision-recall curves, and numerical metrics.
    
    Attributes
    ----------
    metrics : dict
        Dictionary containing computed detection metrics
    labels : List[str]
        Class labels for confusion matrix visualization
    
    Examples
    --------
    >>> manager = DetectionMetricsManager('gt.json', 'preds.json')
    >>> result = manager.calculate_metrics()
    >>> result.plot_confusion_matrix('confusion.png')
    >>> result.export('metrics.json')
    """
    
    def __init__(self, metrics: dict, labels: List[str]):
        """
        Initialize MetricsResult with computed metrics and class labels.
        
        Parameters
        ----------
        metrics : dict
            Dictionary containing computed detection metrics including:
            - confusion_matrix: Binary confusion matrix
            - confusion_matrix_multiclass: Multiclass confusion matrix
            - pr_curves: Precision-Recall curve data
            - Per-class and global metrics
        labels : List[str]
            Class labels for confusion matrix visualization (including background if applicable)
        """
        self.metrics = metrics
        self.labels = labels

    def plot_confusion_matrix(self, path: str = 'confusion_matrix.png', 
                             background_class: bool = False) -> None:
        """
        Plot and save a confusion matrix visualization.
        
        Parameters
        ----------
        path : str, optional
            Output file path for the confusion matrix image, by default 'confusion_matrix.png'
        background_class : bool, optional
            Whether to include background class in the visualization, by default False
            
        Examples
        --------
        >>> result.plot_confusion_matrix('output/confusion.png')
        >>> result.plot_confusion_matrix(background_class=True)
        """
        save_confusion_matrix(
            self.metrics['confusion_matrix_multiclass'], 
            self.labels, 
            path, 
            background_class
        )

    def plot_pr_curves(self, output_path: Optional[str] = 'pr_curves.png', 
                      show: bool = False, dpi: int = 100) -> Optional[plt.Figure]:
        """
        Plot and save Precision-Recall curves.
        
        Parameters
        ----------
        output_path : Optional[str], optional
            Output file path for the PR curve image, by default 'pr_curves.png'
            If None, returns the figure without saving
        show : bool, optional
            Whether to display the plot interactively, by default False
        dpi : int, optional
            Image resolution in dots per inch, by default 100
            
        Returns
        -------
        Optional[plt.Figure]
            Matplotlib figure object if show=False, else None
            
        Examples
        --------
        >>> # Save PR curve to file without displaying
        >>> result.plot_pr_curves('output/pr.png')
        
        >>> # Display interactively without saving
        >>> result.plot_pr_curves(output_path=None, show=True)
        """
        return plot_pr_curves(
            self.metrics.get('pr_curves', {}), 
            output_path, 
            show, 
            dpi
        )

    def export(self, format: str = 'json', path: str = '.') -> None:
        """
        Export metrics to a file in the specified format.
        
        Parameters
        ----------
        format : str, optional
            Output file format (currently only 'json' supported), by default 'json'
        path : str, optional
            Output directory path, by default current directory ('.')
            
        Raises
        ------
        ValueError
            If unsupported format is requested
            
        Examples
        --------
        >>> # Export to JSON in current directory
        >>> result.export()
        
        >>> # Export to specific directory
        >>> result.export(path='output/metrics')
        """
        export_metrics(self.metrics, path, format)