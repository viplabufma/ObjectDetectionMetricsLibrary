from typing import List, Optional

from matplotlib import pyplot as plt
from .visualization import save_confusion_matrix, plot_pr_curves, export_metrics

class MetricsResult:
    """Encapsulates computed detection metrics with methods for visualization and export."""
    def __init__(self, metrics: dict, labels: List[str]):
        self.metrics = metrics
        self.labels = labels

    def plot_confusion_matrix(self, path: str = 'confusion_matrix.png', background_class: bool = False) -> None:
        """Plot and save the confusion matrix."""
        save_confusion_matrix(self.metrics['confusion_matrix_multiclass'], self.labels, path, background_class)

    def plot_pr_curves(self, output_path: Optional[str] = '.pr_curves.png', show: bool = False, dpi: int = 100) -> Optional[plt.Figure]:
        """Plot and save Precision-Recall curves."""
        return plot_pr_curves(self.metrics.get('pr_curves', {}), output_path, show, dpi)

    def export(self, format: str = 'json', path: str = '.') -> None:
        """Export metrics to a file in the specified format."""
        export_metrics(self.metrics, path, format)